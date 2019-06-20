from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import pickle

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw')

class PreProcessor(object):
    def __init__(self):
        nltk.download('wordnet')
        from pyspark.sql import SQLContext
        self.spark = SparkSession.builder.master("local").appName("preprocess").getOrCreate()

        sql_context = SQLContext(self.spark.sparkContext)
        sql_context.clearCache()


    def load_note_events(self,filePath,tempTable1="noteevents",tempTable2="noteevents2"):
        event_struct = StructType([StructField("row_id", IntegerType(), True),
                                StructField("subject_id", IntegerType(), True),
                                StructField("hadm_id", IntegerType(), True),
                                StructField("chartdate", DateType(), True),
                                StructField("category", StringType(), True),
                                StructField("description", StringType(), True),
                                StructField("cgid", IntegerType(), True),
                                StructField("iserror", IntegerType(), True),
                                StructField("text", StringType(), True)])
        df_event = self.spark.read.csv(filePath,
                               header=True,
                               schema=event_struct)
        df_event.registerTempTable(tempTable1)
        df_event.filter(df_event.category == "Discharge summary")\
            .filter(df_event.subject_id < 500)\
            .registerTempTable(tempTable2)
        print("read event notes done")
        return df_event

    def load_diagnose(self,filePath,tempTable="diagnoses_icd_m"):
        diag_struct = StructType([StructField("ROW_ID", IntegerType(), True),
                                  StructField("SUBJECT_ID", IntegerType(), True),
                                  StructField("HADM_ID", IntegerType(), True),
                                  StructField("SEQ_NUM", IntegerType(), True),
                                  StructField("ICD9_CODE", StringType(), True)])
        df_diag_m = self.spark.read.csv(filePath,
                                   header=True,
                                   schema=diag_struct) \
            .selectExpr("ROW_ID as row_id",
                        "SUBJECT_ID as subject_id",
                        "HADM_ID as hadm_id",
                        "SEQ_NUM as seq_num",
                        "ICD9_CODE as icd9_code")
        df_diag_m.filter(df_diag_m.subject_id < 500)
        # added to filter out categories
        geticd9cat_udf = F.udf(lambda x: str(x)[:3], StringType())
        df_diag_m = df_diag_m.withColumn("icd9_cat", geticd9cat_udf("icd9_code"))
        df_diag_m.registerTempTable("diagnoses_icd_m")
        df_diag_m.cache()

        # one icd to one hadm_id (take the smallest seq number as primary)
        diag_o_rdd = df_diag_m.rdd.sortBy(lambda x: (x.hadm_id, x.subject_id, x.seq_num)) \
            .groupBy(lambda x: x.hadm_id) \
            .mapValues(list) \
            .reduceByKey(lambda x, y: x if x.seq_num < y.seq_num else y) \
            .map(lambda d: d[1][0])
        df_diag_o = self.spark.createDataFrame(diag_o_rdd)
        df_diag_o.registerTempTable("diagnoses_icd_o")
        df_diag_o.cache()
        print("read diagnose done")

    def load_icd9_scores(self):
        # get hadm_id list in noteevents
        self.df_hadm_id_list = self.spark.sql("""
        SELECT DISTINCT hadm_id FROM noteevents2 where subject_id <500
        """)
        self.df_hadm_id_list.registerTempTable("hadm_id_list")
        self.df_hadm_id_list.cache()

        self.df_diag_m2 = self.spark.sql("""
        SELECT row_id, subject_id, diagnoses_icd_m.hadm_id AS hadm_id,
        seq_num, icd9_code, icd9_cat
        FROM diagnoses_icd_m JOIN hadm_id_list
        ON diagnoses_icd_m.hadm_id = hadm_id_list.hadm_id
        where diagnoses_icd_m.subject_id <500
        """)
        self.df_diag_m2.registerTempTable("diagnoses_icd_m2")
        self.df_diag_m2.cache()

        self.icd9code_score_hadm = self.spark.sql("""
        SELECT icd9_code, COUNT(DISTINCT hadm_id) AS score
        FROM diagnoses_icd_m2
        GROUP BY icd9_code
        """).rdd.cache()

        self.icd9cat_score_hadm = self.spark.sql("""
        SELECT icd9_cat AS icd9_code, COUNT(DISTINCT hadm_id) AS score
        FROM diagnoses_icd_m2
        GROUP BY icd9_cat
        """).rdd.cache()

    def get_id_to_topicd9(self,id_type, icdcode, topX):
        if id_type == "hadm_id" and icdcode:
            icd9_score = self.icd9code_score_hadm
        elif id_type == "hadm_id" and not icdcode:
            icd9_score = self.icd9cat_score_hadm
        else:  # default
            icd9_score = self.icd9code_score_hadm

        icd9_topX2 = [i.icd9_code for i in icd9_score.takeOrdered(topX, key=lambda x: -x.score)]
        if not icdcode:
            icd9_topX2 = ['c' + str(i) for i in icd9_topX2]
        else:
            icd9_topX2 = [str(i) for i in icd9_topX2]
        icd9_topX = set(icd9_topX2)

        id_to_topicd9 = self.df_diag_m2.rdd \
            .map(lambda x: (
        x.hadm_id if id_type == "hadm_id" else x.subject_id, x.icd9_code if icdcode else 'c' + str(x.icd9_cat))) \
            .groupByKey() \
            .mapValues(lambda x: set(x) & icd9_topX) \
            .filter(lambda x: x[1])

        return id_to_topicd9, list(icd9_topX2)

    def get_id_to_texticd9(self,df_ne,id_type, topX, stopwords=[]):

        def sparse2vec(mapper, data):
            out = [0] * len(mapper)
            if data != None:
                for i in data:
                    out[mapper[i]] = 1
            return out

        def remstopwords(text,lemmatizer):
            text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
            text = re.sub('<[^>]*>', '', text)
            text = re.sub('[\W]+', ' ', text.lower())
            text = re.sub(" \d+", " ", text)
            return " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text) if w not in stopwords])

        id_to_topicd9code, topicd9code = self.get_id_to_topicd9(id_type, True, topX)
        id_to_topicd9cat, topicd9cat = self.get_id_to_topicd9(id_type, False, topX)
        topX2 = 2 * topX
        topicd9 = topicd9code + topicd9cat
        mapper = dict(zip(topicd9, range(topX2)))


        id_to_topicd9 = id_to_topicd9code.fullOuterJoin(id_to_topicd9cat) \
            .map(lambda id_code_cat: (id_code_cat[0], \
                                      (id_code_cat[1][0] if id_code_cat[1][0] else set()) | \
                                      (id_code_cat[1][1] if id_code_cat[1][1] else set())))

        lemmatizer = WordNetLemmatizer()

        event_topX = df_ne.rdd \
            .filter(lambda x: x.category == "Discharge summary") \
            .map(lambda x: (x.hadm_id if id_type == "hadm_id" else x.subject_id, x.text)) \
            .groupByKey() \
            .mapValues(lambda x: " ".join(x)) \
            .leftOuterJoin(id_to_topicd9) \
            .map(lambda id_text_icd9: \
                     [id_text_icd9[0]] + sparse2vec(mapper, id_text_icd9[1][1]) + \
                     [remstopwords(id_text_icd9[1][0],lemmatizer)])

        return self.spark.createDataFrame(event_topX, ["id"] + topicd9 + ["text"]), topicd9

    def get_icd9_codes(self):
        icd9_codes = self.spark.sql("""
            SELECT DISTINCT icd9_code FROM diagnoses_icd_m2
            """).rdd.map(lambda x: x.icd9_code).collect()
        icd9_codes = [str(i).lower() for i in icd9_codes]

        return icd9_codes

    def remove_stop_words(self,df_ne,icd9_codes,topK):
        t0 = time.time()
        STOPWORDS_WORD2VEC = stopwords.words('english') + icd9_codes
        self.spark.sparkContext.broadcast(STOPWORDS_WORD2VEC)
        #print(STOPWORDS_WORD2VEC)
        df_id2texticd9, topicd9 = self.get_id_to_texticd9(df_ne,"hadm_id", topK, stopwords=STOPWORDS_WORD2VEC)
        #df_id2texticd9 = df_id2texticd9.limit(1000)
        df_id2texticd9.coalesce(1).write.csv("./data/DATA_HADM_CLEANED", header=True)
        df_id2texticd9.cache()

        print(topicd9)
        #print(df_id2texticd9.count())
        print(time.time() - t0)
        df_id2texticd9.show()

        return topicd9


def main():
    preprocessor = PreProcessor()

    df_ne = preprocessor.load_note_events("./data/NOTEEVENTS-2.csv")
    print(df_ne)
    preprocessor.load_diagnose("./data/DIAGNOSES_ICD.csv")

    preprocessor.load_icd9_scores()

    icd9_codes = preprocessor.get_icd9_codes()

    pickle.dump(icd9_codes, open("./data/ICD9CODES.p", "wb"))

    topicd9 = preprocessor.remove_stop_words(df_ne,icd9_codes,50)

    print(topicd9[:10])
    pickle.dump(topicd9[:10], open("./data/ICD9CODES_TOP10.p", "wb"))
    print(topicd9[:50])
    pickle.dump(topicd9[:50], open("./data/ICD9CODES_TOP50.p", "wb"))
    print(topicd9[50:60])
    pickle.dump(topicd9[50:60], open("./data/ICD9CAT_TOP10.p", "wb"))
    print(topicd9[50:])
    pickle.dump(topicd9[50:], open("./data/ICD9CAT_TOP50.p", "wb"))


if __name__ =="__main__":
    main()