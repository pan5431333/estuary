//package estuary.demo
//
//import breeze.stats.{mean, stddev}
//import estuary.components.layers.{ReluLayer, SoftmaxLayer}
//import estuary.components.optimizer.AdamOptimizer
//import estuary.helper.GasCensorDataHelper
//import estuary.model.{NNModel, ModelLike}
//import estuary.support.NormalizeUtils
//
///**
//  * Created by mengpan on 2017/8/15.
//  */
//object ClassFourCompoundNeuralNetworkDemo extends App {
//  // Dataset Download Website: http://archive.ics.uci.edu/ml/machine-learning-databases/00224/
//  //加载Gas Censor的数据集
//  //D:\Users\m_pan\Downloads\Dataset\Dataset\\
//  ///Users/mengpan/Downloads/Dataset/
//  val data = GasCensorDataHelper.getAllData("/Users/mengpan/Downloads/Dataset/")
//  //    val data = CatDataHelper.getAllCatData
//
//  //归一化数据特征矩阵
//  val normalizedCatData = NormalizeUtils.normalizeBy(data) { col =>
//    (col - mean(col)) / stddev(col)
//  }
//
//  //获取training set和test set
//  val (training, test) = normalizedCatData.split(0.8)
//
//  //分别获取训练集和测试集的feature和label
//  val trainingFeature = training.getFeatureAsMatrix
//  val trainingLabel = training.getLabelAsVector.map(_.toInt)
//  val testFeature = test.getFeatureAsMatrix
//  val testLabel = test.getLabelAsVector.map(_.toInt)
//
//  val hiddenLayers = List(
//    ReluLayer(numHiddenUnits = 128),
//    ReluLayer(numHiddenUnits = 64))
//  val outputLayer = SoftmaxLayer(6)
//  val nnModel = new NNModel(hiddenLayers, outputLayer)
//
//  //  //Test for performance improved by distributed algorithms
//  //  val adamTime = ModelLike.evaluationTime(nnModel.train(trainingFeature, trainingLabel, AkkaAdamOptimizer(iteration = 30)))
//  //  val distributedAdamTime = ModelLike.evaluationTime(nnModel.train(trainingFeature, trainingLabel, DistributedAdamOptimizer(iteration = 30)))
//  //  println("adamTime: " + adamTime + "ms")
//  //  println("distributedAdamTime: " + distributedAdamTime + "ms")
//
//  //用训练集的数据训练算法
//
//  //  The train accuracy of this model is: 0.994608195542775
//  //  The test accuracy of this model is: 0.6409058231488138
////  val trainedModel = nnModel.train(trainingFeature, trainingLabel, AdamOptimizer())
//
//  //  The train accuracy of this model is: 0.9966750539180446
//  //  The test accuracy of this model is: 0.6229331416247305
//  //  val trainedModel = nnModel.train(trainingFeature, trainingLabel, DistributedAdamOptimizer(iteration = 66, nTasks = 4))
//
//  //测试算法获得算法优劣指标
//  val yPredicted = trainedModel.predictToVector(testFeature)
//  val trainYPredicted = trainedModel.predictToVector(trainingFeature)
//
//  val testAccuracy = ModelLike.accuracy(testLabel, yPredicted)
//  val trainAccuracy = ModelLike.accuracy(trainingLabel, trainYPredicted)
//
//  println("\n The train accuracy of this model is: " + trainAccuracy)
//  println("\n The test accuracy of this model is: " + testAccuracy)
//
//  //对算法的训练过程中cost与迭代次数变化关系进行画图
//  trainedModel.plotCostHistory()
//}
