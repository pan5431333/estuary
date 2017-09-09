package org.mengpan.deeplearning.demo

import breeze.stats.{mean, stddev}
import org.mengpan.deeplearning.components.initializer.HeInitializer
import org.mengpan.deeplearning.components.layers.{DropoutLayer, ReluLayer, SigmoidLayer}
import org.mengpan.deeplearning.components.regularizer.L2Regularizer
import org.mengpan.deeplearning.data.GasCensor
import org.mengpan.deeplearning.helper.{CatDataHelper, DlCollection, GasCensorDataHelper}
import org.mengpan.deeplearning.model.{Model, NeuralNetworkModel}
import org.mengpan.deeplearning.utils.{MyDict, NormalizeUtils, PlotUtils}

/**
  * Created by mengpan on 2017/8/15.
  */
object ClassFourCompoundNeuralNetworkDemo extends App{
  // Dataset Download Website: http://archive.ics.uci.edu/ml/machine-learning-databases/00224/
  //加载Gas Censor的数据集
  val data: DlCollection[GasCensor] = GasCensorDataHelper.getAllData
//  val data = CatDataHelper.getAllCatData

  //归一化数据特征矩阵
  val normalizedCatData = NormalizeUtils.normalizeBy(data){col =>
    (col - mean(col)) / stddev(col)
  }

  //获取training set和test set
  val (training, test) = normalizedCatData.split(0.02)

  //分别获取训练集和测试集的feature和label
  val trainingFeature = training.getFeatureAsMatrix
  val trainingLabel = training.getLabelAsVector
  val testFeature = test.getFeatureAsMatrix
  val testLabel = test.getLabelAsVector

  //初始化算法模型
  val nnModel: Model = new NeuralNetworkModel()
    .setWeightsInitializer(HeInitializer)
    .setRegularizer(new L2Regularizer().setLambda(0.0))
    .setHiddenLayerStructure(List(
      new ReluLayer().setNumHiddenUnits(300),
      new DropoutLayer().setDropoutRate(0.0),
      new ReluLayer().setNumHiddenUnits(100),
      new DropoutLayer().setDropoutRate(0.0)
    ))
    .setOutputLayerStructure(new SigmoidLayer().setNumHiddenUnits(1))
    .setLearningRate(0.01)
    .setIterationTime(3000)

  //用训练集的数据训练算法
  val trainedModel: Model = nnModel.train(trainingFeature, trainingLabel)

  //测试算法获得算法优劣指标
  val yPredicted = trainedModel.predict(testFeature)
  val trainYPredicted = trainedModel.predict(trainingFeature)

  val testAccuracy = trainedModel.accuracy(testLabel, yPredicted)
  val trainAccuracy = trainedModel.accuracy(trainingLabel, trainYPredicted)
  println("\n The train accuracy of this model is: " + trainAccuracy)
  println("\n The test accuracy of this model is: " + testAccuracy)

  //对算法的训练过程中cost与迭代次数变化关系进行画图
  val costHistory = trainedModel.getCostHistory
  PlotUtils.plotCostHistory(costHistory)
}
