package org.mengpan.deeplearning.model
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components._
import org.mengpan.deeplearning.utils.MyDict
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}

/**
  * Created by mengpan on 2017/9/5.
  */
class CompoundNeuralNetworkModel extends NeuralNetworkModel{
  override val logger = Logger.getLogger("CompoundNeuralNetworkModel")

  protected var weightsInitializer: WeightsInitializer = {
    NormalInitializer
  }
  protected var regularizer: Regularizer = {
    VoidRegularizer
  }
  protected var lambda: Double = 0.0

  def setWeightsInitializer(initType: Byte): this.type = {
    this.weightsInitializer = (initType) match {
      case MyDict.INIT_HE => HeInitializer
      case MyDict.INIT_XAIVER => XaiverInitializer
      case _ => logger.info("No init type specified, Normal Initializer used by default!")
        NormalInitializer
    }

    this
  }

  def setRegularizer(reguType: Byte): this.type = {
    this.regularizer = (reguType) match {
      case MyDict.REGULARIZATION_L2 => new L2Regularizer().setLambda(this.lambda)
      case MyDict.REGULARIZATION_L1 => new L1Regularizer().setLambda(this.lambda)
      case _ => VoidRegularizer
    }

    this
  }

  def setLambda(lambda: Double): this.type = {
    this.lambda = lambda
    this.regularizer.setLambda(this.lambda)
    this
  }

  override def train(feature: DenseMatrix[Double], label: DenseVector[Double]):
  CompoundNeuralNetworkModel.this.type = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    //1. Initialize the weights using initializer
    var paramsList: List[(DenseMatrix[Double], DenseVector[Double])] =
      this.weightsInitializer.init(numExamples, inputDim, hiddenLayerStructure, outputLayerStructure)

    //2. Iteration
    (0 until this.iterationTime).foreach{i =>

      //3. forward
      val forwardResList: List[ForwardRes] = forward(feature, paramsList,
        hiddenLayers, outputLayer)

      //4. calculate cost
      val cost = this.regularizer.calCost(forwardResList.last.yCurrent(::, 0), label, paramsList)

      if (i % 100 == 0) {
        logger.info("Cost in " + i + "th time of iteration: " + cost)
      }
      costHistory.put(i, cost)

      //5. backward
      val backwardResList: List[BackwardRes] =
        this.regularizer.backward(feature, label, forwardResList, paramsList, hiddenLayers.toList, outputLayer)

      //6. update parameters
      paramsList = updateParams(paramsList, learningRate, backwardResList, i, cost)
    }

    this.paramsList = paramsList

    this
  }

}
