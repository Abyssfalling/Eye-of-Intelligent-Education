from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# 初始化Spark会话
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.jars", "static/mysql-connector-j-8.3.0/mysql-connector-j-8.3.0.jar") \
    .getOrCreate()



# 数据库连接信息
database_url = "jdbc:mysql://101.227.236.139:3306/education_data"
properties = {"user": "root", "password": "Bigdata2101~", "driver": "com.mysql.cj.jdbc.Driver"}


# 数据文件路径
path_prefix = "data/"  # 需要根据实际情况调整
datasets = {
    "course_chapter": "course_chapter.csv",
    "course_task": "course_task.csv",
    "classroom_member": "classroom_member.csv",
    "classroom_courses": "classroom_courses.csv",
    "log": "log.csv",
    "user_learn_statistics_total": "user_learn_statistics_total.csv",
    "activity_learn_log": "activity_learn_log.csv",
    "testpaper": "testpaper.csv",
    "testpaper_result": "testpaper_result.csv",
    # "教学视频" 对应的数据可能不适合直接以表格形式读取
}

# 读取所有数据表
dataframes = {}
for name, file in datasets.items():
    df = spark.read.csv(f"{path_prefix}{file}", header=True, inferSchema=True)
    dataframes[name] = df

# course_chapter DataFrame写入MySQL
dataframes["course_chapter"].write.jdbc(url=database_url, table="course_chapter", mode="append", properties=properties)

# course_task DataFrame写入MySQL
dataframes["course_task"].write.jdbc(url=database_url, table="course_task", mode="append", properties=properties)

# classroom_member DataFrame写入MySQL
dataframes["classroom_member"].write.jdbc(url=database_url, table="classroom_member", mode="append", properties=properties)

# classroom_courses DataFrame写入MySQL
dataframes["classroom_courses"].write.jdbc(url=database_url, table="classroom_courses", mode="append", properties=properties)

# log DataFrame写入MySQL
dataframes["log"].write.jdbc(url=database_url, table="log", mode="append", properties=properties)

# user_learn_statistics_total DataFrame写入MySQL
dataframes["user_learn_statistics_total"].write.jdbc(url=database_url, table="user_learn_statistics_total", mode="append", properties=properties)

# activity_learn_log DataFrame写入MySQL
dataframes["activity_learn_log"].write.jdbc(url=database_url, table="activity_learn_log", mode="append", properties=properties)

# testpaper DataFrame写入MySQL
dataframes["testpaper"].write.jdbc(url=database_url, table="testpaper", mode="append", properties=properties)

# testpaper_result DataFrame写入MySQL
dataframes["testpaper_result"].write.jdbc(url=database_url, table="testpaper_result", mode="append", properties=properties)
