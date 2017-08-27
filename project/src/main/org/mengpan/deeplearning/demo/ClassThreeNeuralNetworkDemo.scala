package org.mengpan.deeplearning.demo

import breeze.stats.{mean, stddev}
import org.mengpan.deeplearning.data.{Cat, GasCensor}
import org.mengpan.deeplearning.helper.{CatDataHelper, DlCollection, GasCensorDataHelper}
import org.mengpan.deeplearning.model.{Model, NeuralNetworkModel, ShallowNeuralNetworkModel}
import org.mengpan.deeplearning.utils.{MyDict, NormalizeUtils, PlotUtils}

/**
  * Created by mengpan on 2017/8/15.
  */
object ClassThreeNeuralNetworkDemo extends App{
  // Dataset Download Website: http://archive.ics.uci.edu/ml/machine-learning-databases/00224/
  //加载Gas Censor的数据集
  val data: DlCollection[GasCensor] = GasCensorDataHelper.getAllData

  //归一化数据特征矩阵
  val normalizedCatData = NormalizeUtils.normalizeBy(data){col =>
    (col - mean(col)) / stddev(col)
  }


  //获取training set和test set
  val (training, test) = normalizedCatData.split(0.8)

  //分别获取训练集和测试集的feature和label
  val trainingFeature = training.getFeatureAsMatrix
  val trainingLabel = training.getLabelAsVector
  val testFeature = test.getFeatureAsMatrix
  val testLabel = test.getLabelAsVector

  //初始化LR的算法模型
  val nnModel: Model = new NeuralNetworkModel()
    .setHiddenLayerStructure(Map(
      (200, MyDict.ACTIVATION_RELU)
    ))
    .setOutputLayerStructure((1, MyDict.ACTIVATION_SIGMOID))
    .setLearningRate(0.01)
    .setIterationTime(5000)

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
