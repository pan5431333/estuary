package estuary.demo

import breeze.stats.{mean, stddev}
import estuary.components.layers.{DropoutLayer, ReluLayer, SoftmaxLayer}
import estuary.components.optimizer.DistributedAdamOptimizer
import estuary.helper.GasCensorDataHelper
import estuary.model.{FullyConnectedNNModel, Model}
import estuary.utils.NormalizeUtils

/**
  * Created by mengpan on 2017/8/15.
  */
object ClassFourCompoundNeuralNetworkDemo extends App {
  // Dataset Download Website: http://archive.ics.uci.edu/ml/machine-learning-databases/00224/
  //加载Gas Censor的数据集
  val data = GasCensorDataHelper.getAllData("D:\\Users\\m_pan\\Downloads\\Dataset\\Dataset\\")
  //  val data = CatDataHelper.getAllCatData

  //归一化数据特征矩阵
  val normalizedCatData = NormalizeUtils.normalizeBy(data) { col =>
    (col - mean(col)) / stddev(col)
  }

  //获取training set和test set
  val (training, test) = normalizedCatData.split(0.8)

  //分别获取训练集和测试集的feature和label
  val trainingFeature = training.getFeatureAsMatrix
  val trainingLabel = training.getLabelAsVector.map(_.toInt)
  val testFeature = test.getFeatureAsMatrix
  val testLabel = test.getLabelAsVector.map(_.toInt)

  val hiddenLayers = List(
    ReluLayer(numHiddenUnits = 200),
    DropoutLayer(0.5),
    ReluLayer(numHiddenUnits = 200),
    DropoutLayer(0.5),
    ReluLayer(numHiddenUnits = 200),
    DropoutLayer(0.5),
    ReluLayer(numHiddenUnits = 100))
  val outputLayer = SoftmaxLayer()
  val nnModel = new FullyConnectedNNModel(hiddenLayers, outputLayer, 0.01, 50, None)

  //  //Test for performance improved by distributed algorithms
  //  val adamTime = Model.evaluationTime(nnModel.train(trainingFeature, trainingLabel, AdamOptimizer(64)))
  //  val distributedAdamTime = Model.evaluationTime(nnModel.train(trainingFeature, trainingLabel, DistributedAdamOptimizer(64, 4)))
  //  println("adamTime: " + adamTime + "ms")
  //  println("distributedAdamTime: " + distributedAdamTime + "ms")

  //用训练集的数据训练算法
  val trainedModel = nnModel.train(trainingFeature, trainingLabel, DistributedAdamOptimizer())
  //测试算法获得算法优劣指标
  val yPredicted = trainedModel.predict(testFeature)
  val trainYPredicted = trainedModel.predict(trainingFeature)

  val testAccuracy = Model.accuracy(testLabel, yPredicted)
  val trainAccuracy = Model.accuracy(trainingLabel, trainYPredicted)
  println("\n The train accuracy of this model is: " + trainAccuracy)
  println("\n The test accuracy of this model is: " + testAccuracy)

  //对算法的训练过程中cost与迭代次数变化关系进行画图
  trainedModel.plotCostHistory()
}
