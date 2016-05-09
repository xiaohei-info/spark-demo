package info.xiaohei.www.spark.examples.kmeans

import org.apache.spark.mllib.clustering.{KMeansModel, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Copyright © 2016 xiaohei, All Rights Reserved.
  * Email : chubbyjiang@gmail.com
  * Host : xiaohei.info
  * Created : 16/5/9 09:52
  */
object RunKMeans {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("KMeans")
    val sc = new SparkContext(conf)
    //读取数据
    val rawData = sc.textFile("/spark_data/ch05/kddcup.data")
    //根据类别查看统计信息,各个类别下有多少数据
    //catStatsData(rawData)

    val labelsAndData = rawData.map { line =>
      //buffer是一个可变列表
      val buffer = line.split(",").toBuffer
      //下标1-3的元素
      buffer.remove(1, 3)
      //最后一个元素为label
      val label = buffer.remove(buffer.length - 1)
      //转换为Vector
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
    //数据只用到values部分
    val data = labelsAndData.values.cache()
    //第一次训练模型
    //firstKMeans(data, labelsAndData)

    //取不同k值观察模型优劣
    /*(50 to 130 by 10).map { k =>
      (k, clusteringScore(data, k))
    }.foreach(println)*/

    //取最好的k值训练模型
    val kmeans = new KMeans()
    kmeans.setK(1)
    kmeans.setEpsilon(1.0e-6)
    kmeans.setRuns(10)
    val model = kmeans.run(data)
    catLabelCount(labelsAndData, model)
  }

  /**
    * 输出样本数据的统计信息
    **/
  def catStatsData(rawData: RDD[String]): Unit = {
    //根据","分割,只保留最后的类别
    val catStatsData = rawData.map(_.split(",").last)
      //对类别的数目进行统计,并根据统计的数量从小打到排序
      .countByValue().toSeq.sortBy(_._2)
      //转换为从大到小排序
      .reverse
    catStatsData.foreach(println)
  }

  /**
    * 初次训练模型
    **/
  def firstKMeans(data: RDD[Vector]
                  , labelsAndData: RDD[(String, Vector)]): Unit = {
    //训练模型
    val kmeans = new KMeans()
    val model = kmeans.run(data)
    //输出每个族群的点
    model.clusterCenters.foreach(println)
    catLabelCount(labelsAndData, model)
  }

  /**
    * 输出每个族群包含的类别和个数信息
    *
    * @param labelsAndData 含类别信息的数据
    * @param model         KMeans模型
    **/
  def catLabelCount(labelsAndData: RDD[(String, Vector)], model: KMeansModel): Unit = {
    //输出每个聚类中心有哪些类别各有多少个数据
    val clusterLabelCount = labelsAndData.map { case (label, datum) =>
      //为样本数据划分聚类中心
      val cluster = model.predict(datum)
      //返回数据的中心和类别二元组
      (cluster, label)
    }.countByValue()
    //排序之后格式化输出
    clusterLabelCount.toSeq.sorted.foreach { case ((cluster, label), count) =>
      println(f"$cluster\t$label\t$count")
    }
  }

  /**
    * 计算两个向量之间的距离
    *
    * @param a 向量1
    * @param b 向量2
    *          欧式距离:空间上两个点的距离=两个向量相应元素的差的平方和的平方根
    **/
  def distance(a: Vector, b: Vector) = {
    //求平方根
    math.sqrt(
      //将两个向量合并
      a.toArray.zip(b.toArray)
        //两个向量中的每个值相减
        .map(d => d._1 - d._2)
        //相间的值平方
        .map(d => d * d)
        //之后相加
        .sum)
  }

  /**
    * 计算数据点到聚类中心质心的距离
    *
    * @param datum 数据点
    * @param model kmeans模型
    **/
  def distToCentrolid(datum: Vector, model: KMeansModel) = {
    //得到该数据点的聚类中心
    val cluster = model.predict(datum)
    //得到该聚类中心的质心
    val centrolid = model.clusterCenters(cluster)
    //计算距离
    distance(centrolid, datum)
  }

  /**
    * 根据各个数据点到该数据点聚类中心质心的距离来判断该模型优劣
    *
    * @param data 样本数据
    * @param k    k值
    **/
  def clusteringScore(data: RDD[Vector], k: Int) = {
    val kmeans = new KMeans()
    //设置k值
    kmeans.setK(k)
    //设置该k值的聚类次数
    kmeans.setRuns(10)
    //设置迭代过程中,质心的最小移动值,默认为1.0e-4
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(data)
    //计算样本数据到其各自质心的记录的平均值
    data.map { datum =>
      distToCentrolid(datum, model)
    }.mean()
  }
}
