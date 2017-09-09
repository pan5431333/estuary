package org.mengpan.deeplearning.model
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, pow}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.initializer.{HeInitializer, NormalInitializer, WeightsInitializer, XaiverInitializer}
import org.mengpan.deeplearning.components.layers.{DropoutLayer, EmptyLayer, Layer}
import org.mengpan.deeplearning.components.regularizer.{L1Regularizer, L2Regularizer, Regularizer, VoidRegularizer}
import org.mengpan.deeplearning.utils.{DebugUtils, MyDict, ResultUtils}
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}

/**
  * Created by mengpan on 2017/9/5.
  */
class NeuralNetworkModel extends Model{
  override val logger = Logger.getLogger("CompoundNeuralNetworkModel")

  //神经网络超参数
  override var learningRate: Double = 0.01 //学习率，默认0.01
  override var iterationTime: Int = 3000 //迭代次数，默认3000次
  protected var hiddenLayers: List[Layer] = null
  protected var outputLayer: Layer = null
  protected var weightsInitializer: WeightsInitializer = NormalInitializer //初始化方式，默认一般随机初始化（乘以0.01）
  protected var regularizer: Regularizer = VoidRegularizer //正则化方式，默认无正则化

  lazy val allLayers = hiddenLayers ::: outputLayer :: Nil

  //神经网络模型的参数，由(w,b)组成的List
  var paramsList: List[(DenseMatrix[Double], DenseVector[Double])] = null

  type NNParams = List[(DenseMatrix[Double], DenseVector[Double])]

  //以下是一些设定神经网络超参数的setter
  def setHiddenLayerStructure(hiddenLayers: List[Layer]): this.type = {
    if (hiddenLayers.isEmpty) {
      throw new IllegalArgumentException("hidden layer should be at least one layer!")
    }

    //重点需要解释的地方，如果某一层是Dropout Layer，则将其神经元数量设置成与前一层神经元数量相同
    //这里重点关注scanLeft的用法！
    val theHiddenLayer: List[Layer] = hiddenLayers
      .scanLeft[Layer, List[Layer]](EmptyLayer){
      (previousLayer, currentLayer) =>
      if (currentLayer.isInstanceOf[DropoutLayer])
        currentLayer.setNumHiddenUnits(previousLayer.numHiddenUnits)
      else currentLayer
    }

    this.hiddenLayers = theHiddenLayer.tail //取tail的原因是把第一个EmptyLayer去掉
    this
  }

  def setOutputLayerStructure(outputLayer: Layer): this.type = {
    this.outputLayer = outputLayer
    this
  }

  def setWeightsInitializer(initializer: WeightsInitializer): this.type = {
    this.weightsInitializer = initializer
    this
  }

  def setRegularizer(regularizer: Regularizer): this.type = {
    this.regularizer = regularizer
    this
  }


  override def train(feature: DenseMatrix[Double], label: DenseVector[Double]):
  NeuralNetworkModel.this.type = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    //1. initialize parameters
    val initParams = this.weightsInitializer.init(inputDim, this.allLayers)

    this.paramsList =
      (0 until this.iterationTime) //2. iteration
        .foldLeft[NNParams](initParams){
        (previousParams, iterationTime) =>

          //3. forward
          val forwardResList = forward(feature, previousParams)

          //4. calculate cost
          val cost = calCost(label, forwardResList.last.yCurrent(::, 0), previousParams, this.regularizer)

          //record cost history
          if (iterationTime % 100 == 0) {
            logger.info("Cost in " + iterationTime + "th time of iteration: " + cost)
          }
          costHistory.put(iterationTime, cost)

          //5. backward
          val backwardResList = backward(label, forwardResList, previousParams, this.regularizer)

          //6. update parameters
          updateParams(previousParams, this.learningRate, backwardResList, iterationTime, cost)
      }

    this
  }

  override def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {
    val forwardResList: List[ForwardRes] = forwardWithoutDropout(feature, this.paramsList)
    forwardResList.last.yCurrent(::, 0).map{yHat =>
      if (yHat > 0.5) 1.0 else 0.0
    }
  }

  protected def forward(feature: DenseMatrix[Double],
                        params: List[(DenseMatrix[Double],
                          DenseVector[Double])]): List[ForwardRes] = {

    val initForwardRes = ForwardRes(null, null, feature)
    params
      .zip(this.allLayers)
      .scanLeft[ForwardRes, List[ForwardRes]](initForwardRes){
      (previousForwardRes, f) =>
        val yPrevious = previousForwardRes.yCurrent

        val (w, b) = f._1
        val layer = f._2

        layer.forward(yPrevious, w, b)
      }
      .tail
  }

  protected def forwardWithoutDropout(feature: DenseMatrix[Double],
                                      params: List[(DenseMatrix[Double], DenseVector[Double])]):
  List[ForwardRes] = {

    val initForwardRes = ForwardRes(null, null, feature)
    params
      .zip(this.allLayers)
      .scanLeft[ForwardRes, List[ForwardRes]](initForwardRes){
      (previousForwardRes, f) =>
        val yPrevious = previousForwardRes.yCurrent

        val (w, b) = f._1
        val oldLayer = f._2

        val layer =
          if (oldLayer.isInstanceOf[DropoutLayer])
            new DropoutLayer().setNumHiddenUnits(oldLayer.numHiddenUnits).setDropoutRate(0.0)
          else oldLayer

        layer.forward(yPrevious, w, b)
      }
      .tail
  }

  protected def updateParams(paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                             learningrate: Double,
                             backwardResList: List[ResultUtils.BackwardRes],
                             iterationTime: Int,
                             cost: Double): List[(DenseMatrix[Double], DenseVector[Double])] = {
    paramsList
      .zip(backwardResList)
      .zip(this.allLayers)
      .map{f =>
        val layer = f._2
        val (w, b) = f._1._1

        layer match {
          case _: DropoutLayer => (w, b)
          case _ =>
            val backwardRes = f._1._2
            val dw = backwardRes.dWCurrent
            val db = backwardRes.dBCurrent

            logger.debug(DebugUtils.matrixShape(w, "w"))
            logger.debug(DebugUtils.matrixShape(dw, "dw"))

            w :-= dw * learningrate
            b :-= db * learningrate
            (w, b)
        }
      }
  }

  private def calCost(label: DenseVector[Double], predicted: DenseVector[Double],
                      paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                      regularizer: Regularizer): Double = {
    val originalCost = -(label.t * log(predicted + pow(10.0, -9)) + (1.0 - label).t * log(1.0 - predicted + pow(10.0, -9))) / label.length.toDouble
    val reguCost = regularizer.getReguCost(paramsList)

    originalCost + regularizer.lambda * reguCost / label.length.toDouble
  }

  private def backward(label: DenseVector[Double],
                       forwardResList: List[ResultUtils.ForwardRes],
                       paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                       regularizer: Regularizer): List[BackwardRes] = {
    val yPredicted = forwardResList.last.yCurrent(::, 0)
    val numExamples = label.length

    val dYPredicted = -(label /:/ (yPredicted + pow(10.0, -9)) - (1.0 - label) /:/ (1.0 - yPredicted + pow(10.0, -9)))

    val dYHat = DenseMatrix.zeros[Double](numExamples, 1)
    dYHat(::, 0) := dYPredicted

    val initBackwardRes = BackwardRes(dYHat, null, null)

    paramsList
      .zip(this.allLayers)
      .zip(forwardResList)
      .scanRight[BackwardRes, List[BackwardRes]](initBackwardRes){
      (f, previousBackwardRes) =>
        val dYCurrent = previousBackwardRes.dYPrevious

        val (w, b) = f._1._1
        val layer = f._1._2
        val forwardRes = f._2

        val backwardRes = layer.backward(dYCurrent, forwardRes, w, b)

        layer match {
          case _: DropoutLayer => backwardRes
          case _ =>
            new BackwardRes(backwardRes.dYPrevious,
              backwardRes.dWCurrent + regularizer.getReguCostGrad(w, numExamples),
              backwardRes.dBCurrent)
        }
    }
      .dropRight(1)
  }

}
