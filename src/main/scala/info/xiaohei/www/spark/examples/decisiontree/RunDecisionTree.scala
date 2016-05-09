package info.xiaohei.www.spark.examples.decisiontree

import org.apache.spark.mllib.evaluation.{MulticlassMetrics, MultilabelMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{RandomForest, DecisionTree}
import org.apache.spark.mllib.tree.model.{RandomForestModel, DecisionTreeModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Copyright © 2016 xiaohei, All Rights Reserved.
  * Email : chubbyjiang@gmail.com
  * Host : xiaohei.info
  * Created : 16/5/6 11:25
  */
object RunDecisionTree {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("DecisionTree")
    val sc = new SparkContext(conf)
    //读取数据
    val rawData = sc.textFile("/spark_data/covtype.data")
    //转换为为LabeledPoint
    val data = rawData.map { line =>
      val values = line.split(",").map(_.toDouble)
      //init返回除了最后一个元素的所有元素,作为特征向量
      val feature = Vectors.dense(values.init)
      //返回最后一个目标特征,由于决策树的目标特征规定从0开始,而数据是从1开始的,所以要-1
      val label = values.last - 1
      LabeledPoint(label, feature)
    }

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    //第一个决策树模型
    //val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)
    //使用最好的参数组合的决策树模型
    //val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "entropy", 20, 300)
    //构建随机森林
    val model = RandomForest.trainClassifier(trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "entropy", 30, 300)
    val metrics = getMetrics(model, cvData)

    //混淆矩阵和模型准确度
    System.out.println(metrics.confusionMatrix)
    System.out.println(metrics.precision)

    //每个类别对应的准确度
    (0 until 7).map(target => (metrics.precision(target), metrics.recall(target))).foreach(println)



  }

  /**
    * 获得评估指标
    *
    * @param model 决策树模型
    * @param data  用于交叉验证的数据集
    **/
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    //将交叉验证数据集的每个样本的特征向量交给模型预测,并和原本正确的目标特征组成一个tuple
    val predictionsAndLables = data.map { d =>
      (model.predict(d.features), d.label)
    }
    //将结果交给MulticlassMetrics,其可以以不同的方式计算分配器预测的质量
    new MulticlassMetrics(predictionsAndLables)
  }

  /**
    * @param model 随机啥森林模型
    * @param data  用于交叉验证的数据集
    * */
  def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    //将交叉验证数据集的每个样本的特征向量交给模型预测,并和原本正确的目标特征组成一个tuple
    val predictionsAndLables = data.map { d =>
      (model.predict(d.features), d.label)
    }
    //将结果交给MulticlassMetrics,其可以以不同的方式计算分配器预测的质量
    new MulticlassMetrics(predictionsAndLables)
  }

  /**
    * 在训练数据集上得到最好的参数组合
    *
    * @param trainData 训练数据集
    * @param cvData    交叉验证数据集
    **/
  def getBestParam(trainData: RDD[LabeledPoint], cvData: RDD[LabeledPoint]): Unit = {
    val evaluations = for (impurity <- Array("gini", "entropy");
                           depth <- Array(1, 20);
                           bins <- Array(10, 300)) yield {
      val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), impurity, depth, bins)
      val metrics = getMetrics(model, cvData)
      ((impurity, depth, bins), metrics.precision)
    }
    evaluations.sortBy(_._2).reverse.foreach(println)
  }
}
