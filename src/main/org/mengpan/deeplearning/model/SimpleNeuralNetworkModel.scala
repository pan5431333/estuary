package org.mengpan.deeplearning.model
import java.util

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, pow}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.layers.Layer
import org.mengpan.deeplearning.utils.{DebugUtils, LayerUtils, ResultUtils}
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}

import scala.collection.mutable

/**
  * Created by mengpan on 2017/8/26.
  */
class SimpleNeuralNetworkModel extends Model{
  //记录log
  val logger = Logger.getLogger("NeuralNetworkModel")

  //神经网络的四个超参数
  override var learningRate: Double = _
  override var iterationTime: Int = _
  var hiddenLayerStructure: Map[Int, Byte] = _
  var outputLayerStructure: (Int, Byte) = _

  //记录每一次迭代cost变化的历史数据
  override val costHistory: mutable.TreeMap[Int, Double] = new mutable.TreeMap[Int, Double]()

  //神经网络模型的参数
  var paramsList: List[(DenseMatrix[Double], DenseVector[Double])] = _

  //神经网络的隐含层与输出层的结构，根据hiddenLayerStructure与outputLayerStructure两个超参数得到
  protected var hiddenLayers: Seq[Layer] = _
  protected var outputLayer: Layer = _

  def setHiddenLayerStructure(hiddenLayerStructure: Map[Int, Byte]): this.type = {
    if (hiddenLayerStructure.isEmpty) {
      throw new Exception("hidden layer should be at least one layer!")
    }

    this.hiddenLayerStructure = hiddenLayerStructure
    this.hiddenLayers = getHiddenLayers(this.hiddenLayerStructure)
    this
  }

  def setOutputLayerStructure(outputLayerStructure: (Int, Byte)): this.type = {
    this.outputLayerStructure = outputLayerStructure
    this.outputLayer = getOutputLayer(this.outputLayerStructure)
    this
  }

  override def train(feature: DenseMatrix[Double], label: DenseVector[Double]): SimpleNeuralNetworkModel.this.type = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    logger.debug("hidden layers: " + hiddenLayers)
    logger.debug("output layer: " + outputLayer)

    //随机初始化模型参数
    var paramsList: List[(DenseMatrix[Double], DenseVector[Double])] =
      initializeParams(numExamples, inputDim, hiddenLayers, outputLayer)

    (0 until this.iterationTime).foreach{i =>
      val forwardResList: List[ForwardRes] = forward(feature, paramsList,
        hiddenLayers, outputLayer)

      logger.debug(forwardResList)

      val cost = calCost(forwardResList.last, label)
      if (i % 100 == 0) {
        logger.info("Cost in " + i + "th time of iteration: " + cost)
      }
      costHistory.put(i, cost)

      val backwardResList: List[BackwardRes] = backward(feature, label, forwardResList,
        paramsList, hiddenLayers, outputLayer)

      logger.debug(backwardResList)

      paramsList = updateParams(paramsList, this.learningRate, backwardResList, i, cost)
    }

    this.paramsList = paramsList
    this
  }

  override def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {
    val forwardResList: List[ForwardRes] = forward(feature, this.paramsList,
      this.hiddenLayers, this.outputLayer)
    forwardResList.last.yCurrent(::, 0).map{yHat =>
      if (yHat > 0.5) 1.0 else 0.0
    }
  }

  private def getHiddenLayers(hiddenLayerStructure: Map[Int, Byte]): Seq[Layer] = {
    hiddenLayerStructure.map{structure =>
      getLayerByStructure(structure)
    }.toList
  }

  private def getOutputLayer(structure: (Int, Byte)): Layer = {
    getLayerByStructure(structure)
  }

  private def getLayerByStructure(structure: (Int, Byte)): Layer = {
    val numHiddenUnits = structure._1
    val activationType = structure._2

    val layer: Layer = LayerUtils.getLayerByActivationType(activationType)
      .setNumHiddenUnits(numHiddenUnits)
    layer
  }

  private def initializeParams(numExamples: Int, inputDim: Int,
                               hiddenLayers: Seq[Layer], outputLayer: Layer):
  List[(DenseMatrix[Double], DenseVector[Double])] = {

    /*
     *把输入层，隐含层，输出层的神经元个数组合成一个Vector
     *如inputDim=3，outputDim=1，hiddenDim=(3, 3, 2)，则layersDim=(3, 3, 3, 2, 1)
     *两个List的操作符，A.::(b)为在A前面加上元素B，A.:+(B)为在A的后面加上元素B
     *这里使用Vector存储layersDim，因为Vector为indexed sequence，访问任意位置的元素时间相同
    */
    val layersDim = hiddenLayers.map(_.numHiddenUnits)
      .toList
      .::(inputDim)
      .:+(outputLayer.numHiddenUnits)
      .toVector

    val numLayers = layersDim.length

    /*
     *W(l)的维度为(layersDim(l-1), layersDim(l))
     *b(l)的维度为(layersDim(l), )
     *注意随机初始化的数值在0-1之间，为保证模型稳定性，需在w和b后面*0.01
    */
    (1 until numLayers).map{i =>
      val w = DenseMatrix.rand[Double](layersDim(i-1), layersDim(i)) * 0.01
      val b = DenseVector.rand[Double](layersDim(i)) * 0.01
      (w, b)
    }.toList
  }

  protected def forward(feature: DenseMatrix[Double],
                      params: List[(DenseMatrix[Double],
                        DenseVector[Double])],
                      hiddenLayers: Seq[Layer],
                      outputLayer: Layer): List[ForwardRes] = {
    var yi = feature

    /*
     *这里注意Scala中zip的用法。假设A=List(1, 2, 3), B=List(3, 4), 则
     * A.zip(B) 为 List((1, 3), (2, 4))
     * 复习：A.:+(b)的作用是在A后面加上b元素，注意因为immutable，实际上是生成了一个新对象
     */
    params.zip(hiddenLayers.:+(outputLayer))
      .map{f =>
        val w = f._1._1
        val b = f._1._2
        val layer = f._2

        //forward方法需要yPrevious, w, b三个参数
        val forwardRes = layer.forward(yi, w, b)
        yi = forwardRes.yCurrent

        forwardRes
      }
  }

  private def calCost(res: ResultUtils.ForwardRes, label: DenseVector[Double]):
  Double = {
    val yHat = res.yCurrent(::, 0)

    //在log函数内加上pow(10.0, -9)，防止出现log(0)从而NaN的情况
    -(label.t * log(yHat + pow(10.0, -9)) + (1.0 - label).t * log(1.0 - yHat + pow(10.0, -9))) / label.length.toDouble
  }


  private def backward(feature: DenseMatrix[Double], label: DenseVector[Double],
                       forwardResList: List[ResultUtils.ForwardRes],
                       paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                       hiddenLayers: Seq[Layer], outputLayer: Layer):
  List[BackwardRes] = {
    val yHat = forwardResList.last.yCurrent(::, 0)

    //+ pow(10.0, -9)防止出现被除数为0，NaN的情况
    val dYL = -(label /:/ (yHat + pow(10.0, -9)) - (1.0 - label) /:/ (1.0 - yHat + pow(10.0, -9)))
    var dYCurrent = DenseMatrix.zeros[Double](feature.rows, 1)
    dYCurrent(::, 0) := dYL

    paramsList
      .zip(forwardResList)
      .zip(hiddenLayers.:+(outputLayer))
      .reverse
      .map{f =>
        val w = f._1._1._1
        val b = f._1._1._2
        val forwardRes = f._1._2
        val layer = f._2

        logger.debug(DebugUtils.matrixShape(w, "w"))
        logger.debug(layer)

        /*
         *backward方法需要dYCurrent, forwardRes, w, b四个参数
         * 其中，forwardRes中有用的为：yPrevious(计算dW)，zCurrent（计算dZCurrent）
         */
        val backwardRes = layer.backward(dYCurrent, forwardRes, w, b)
        dYCurrent = backwardRes.dYPrevious
        backwardRes
      }
      .reverse
  }

  protected def updateParams(paramsList: List[(DenseMatrix[Double], DenseVector[Double])],
                           learningrate: Double,
                           backwardResList: List[ResultUtils.BackwardRes],
                           iterationTime: Int, cost: Double): List[(DenseMatrix[Double], DenseVector[Double])] = {
    paramsList.zip(backwardResList)
      .map{f =>
        val w = f._1._1
        val b = f._1._2
        val backwardRes = f._2
        val dw = backwardRes.dWCurrent
        val db = backwardRes.dBCurrent

        logger.debug(DebugUtils.matrixShape(w, "w"))
        logger.debug(DebugUtils.matrixShape(dw, "dw"))

        var adjustedLearningRate = this.learningRate

        //如果cost出现NaN则把学习率降低100倍
        adjustedLearningRate = if (cost.isNaN) adjustedLearningRate/100 else adjustedLearningRate

        w :-= dw * learningrate
        b :-= db * learningrate
        (w, b)
      }
  }

}
