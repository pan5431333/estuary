package estuary.demo

import breeze.stats.{mean, stddev}
import estuary.components.initializer.HeInitializer
import estuary.components.layers.{DropoutLayer, ReluLayer, SoftmaxLayer}
import estuary.components.optimizer.AdamOptimizer
import estuary.components.regularizer.VoidRegularizer
import estuary.helper.{CatDataHelper, GasCensorDataHelper}
import estuary.model.{Model, NeuralNetworkModel}
import estuary.utils.{NormalizeUtils, PlotUtils}

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
  val trainingLabel = training.getLabelAsVector
  val testFeature = test.getFeatureAsMatrix
  val testLabel = test.getLabelAsVector

  //初始化算法模型
  val nnModel: Model = new NeuralNetworkModel()
    .setWeightsInitializer(HeInitializer)
    .setRegularizer(VoidRegularizer)
    .setOptimizer(AdamOptimizer(miniBatchSize = 128))
    .setHiddenLayerStructure(
      ReluLayer(numHiddenUnits = 200, batchNorm = false),
      DropoutLayer(dropoutRate = 0.1),
      ReluLayer(numHiddenUnits = 100, batchNorm = false),
      DropoutLayer(dropoutRate = 0.1)
    )
    .setOutputLayerStructure(SoftmaxLayer(batchNorm = false))
    .setLearningRate(0.0001)
    .setIterationTime(20)

  //API 2nd version
  //  val nnModel = NeuralNetworkModel(List(ReluLayer(200), ReluLayer(100)), SigmoidLayer(1))

  //用训练集的数据训练算法
  val trainedModel = nnModel.train(trainingFeature, trainingLabel)

  //  trainedModel.asInstanceOf[NeuralNetworkModel].setOutputLayerStructure(SoftmaxLayer(false))
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
