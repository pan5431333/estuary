package org.mengpan.deeplearning.model
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{log, pow}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.caseclasses.AdamOptimizationParams
import org.mengpan.deeplearning.components.initializer.{HeInitializer, NormalInitializer, WeightsInitializer, XaiverInitializer}
import org.mengpan.deeplearning.components.layers.{DropoutLayer, EmptyLayer, Layer}
import org.mengpan.deeplearning.components.optimizer._
import org.mengpan.deeplearning.components.regularizer.{L1Regularizer, L2Regularizer, Regularizer, VoidRegularizer}
import org.mengpan.deeplearning.utils.{DebugUtils, MyDict, ResultUtils}
import org.mengpan.deeplearning.utils.ResultUtils.{BackwardRes, ForwardRes}

/**
  * Created by meng.pan on 2017/9/5.
  */
class NeuralNetworkModel extends Model{
  override val logger = Logger.getLogger(this.getClass)

  //Hyperparameters of neural nets
  override var learningRate: Double = 0.01
  override var iterationTime: Int = 3000
  protected var hiddenLayers: List[Layer] = null
  protected var outputLayer: Layer = null
  protected var weightsInitializer: WeightsInitializer = HeInitializer //initialization methods, see also HeInitializer, XaiverInitializer
  protected var regularizer: Regularizer = VoidRegularizer //VoidRegularizer: No regularization. see also L1Regularizer, L2Regularizer
  protected var optimizer: Optimizer = AdamOptimizer() //AdamOptimizer, see also GDOptimizer: Batch Gradient Descent, SGDOptimizer,

  lazy val allLayers = hiddenLayers ::: outputLayer :: Nil //concat hidden layers and output layer

  type NNParams = List[(DenseMatrix[Double], DenseVector[Double])] //Neural Network Model parameters, consisting of List[(w, b)]
  var paramsList: NNParams = null

  def setHiddenLayerStructure(hiddenLayers: List[Layer]): this.type = {
    if (hiddenLayers.isEmpty) {
      throw new IllegalArgumentException("hidden layer should be at least one layer!")
    }

    //if a layer is Dropout Layer，then set its number of hidden units to be same as its previous layer
    val theHiddenLayer: List[Layer] = hiddenLayers
      .scanLeft[Layer, List[Layer]](EmptyLayer){
      (previousLayer, currentLayer) =>
      if (currentLayer.isInstanceOf[DropoutLayer])
        currentLayer.setNumHiddenUnits(previousLayer.numHiddenUnits)
      else currentLayer
    }

    this.hiddenLayers = theHiddenLayer.tail //drop the first EmptyLayer in the list
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

  def setOptimizer(optimizer: Optimizer): this.type = {
    this.optimizer = optimizer
    this
  }


  /**
    * Train the model.
    * If the optimizer chosen is AdamOptimizer, then use method trainWithMomentumAndAdam.
    * else use trainWithoutMomentum.
    * @param feature Feature matrix whose shape is (numTrainingExamples, inputFeatureDimension)
    * @param label Label vector whose length is numTrainingExamples
    * @todo when any other optimizer is created, the case MUST be added here
    * @return A trained model whose model parameters have been updated
    */
  override def train(feature: DenseMatrix[Double],
                     label: DenseVector[Double]): NeuralNetworkModel.this.type = {

    this.paramsList = this.optimizer match {
      case op: AdamOptimizer => trainByAdam(feature, label, op)
      case op: NonHeuristic => trainWithoutMomentum(feature, label, op)
    }

    this
  }

  /**
    * Do predictions. Note prediction is done without the dropout layer.
    * @param feature Feature matrix with shape (numExamples, inputFeatureDim)
    * @return Predicted labels
    */
  override def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {

    assert(this.paramsList != null, "Model has not been trained, please train it first")
    assert(this.paramsList.head._1.rows == feature.cols, "Prediction's features' dimension is not as the same as trained features'")

    val forwardResList: List[ForwardRes] = forwardWithoutDropout(feature, this.paramsList)
    forwardResList.last.yCurrent(::, 0).map{yHat =>
      if (yHat > 0.5) 1.0 else 0.0
    }
  }

  /**
    * Forward propagation of all layers.
    * @param feature Feature matrix with shape (numExamples, inputFeatureDim)
    * @param params Model parameters of the form List((w, b))
    *               where w is of shape (d(L-1), d(L)) and b of (d(L)) for Lth layer
    *               where d(L) is the number of hidden units in Lth layer.
    * @return List of ForwardRes consisting of (yPrevious, zCurrent, yCurrent)
    *         where yPrevious: the output from previous layer, say L-1 th layer, of shape (numExamples, d(L-1))
    *         zCurrent = yPrevious * w(L) + DenseVector.ones[Double](numExamples) * b(L).t
    *         yCurrent = g(zCurrent) where g is the activation function in Lth layer.
    */
  private def forward(feature: DenseMatrix[Double],
                      params: List[(DenseMatrix[Double], DenseVector[Double])]): List[ForwardRes] = {

    val initForwardRes = ForwardRes(null, null, feature) // constructing a ForwardRes instance for input layer

    params
      .zip(this.allLayers)
      .scanLeft[ForwardRes, List[ForwardRes]](initForwardRes){
      (previousForwardRes, f) =>
        val yPrevious = previousForwardRes.yCurrent

        val (w, b) = f._1
        val layer = f._2

        layer.forward(yPrevious, w, b)
      }
      .tail //Drop the first ForwardRes in input layer
  }

  /**
    * Forward propagation of all layers with setting dropout rate of all Dropout layers as 0.
    * @param feature Feature matrix of shape (numExamples, inputFeatureDim)
    * @param params Model parameters of the form List((w, b))
    *               where w is of shape (d(L-1), d(L)) and b of (d(L)) for Lth layer
    *               where d(L) is the number of hidden units in Lth layer.
    * @return List of ForwardRes consisting of (yPrevious, zCurrent, yCurrent)
    *         where yPrevious: the output from previous layer, say L-1 th layer, of shape (numExamples, d(L-1))
    *         zCurrent = yPrevious * w(L) + DenseVector.ones[Double](numExamples) * b(L).t
    *         yCurrent = g(zCurrent) where g is the activation function in Lth layer.
    */
  private def forwardWithoutDropout(feature: DenseMatrix[Double],
                                    params: NNParams): List[ForwardRes] = {

    val initForwardRes = ForwardRes(null, null, feature) // constructing a ForwardRes instance for input layer

    params
      .zip(this.allLayers)
      .scanLeft[ForwardRes, List[ForwardRes]](initForwardRes){
      (previousForwardRes, f) =>
        val yPrevious = previousForwardRes.yCurrent

        val (w, b) = f._1
        val oldLayer = f._2

        val layer = if (oldLayer.isInstanceOf[DropoutLayer])
                      new DropoutLayer().setNumHiddenUnits(oldLayer.numHiddenUnits).setDropoutRate(0.0) //set dropout rate to 0.0, i.e. turn off dropout.
                    else oldLayer

        layer.forward(yPrevious, w, b)
      }
      .tail // drop the first ForwardRes of input layer
  }

  /**
    * Calculate value of the cost function, consisting the sum of cross-entropy cost and regularization cost.
    * @param label True label vector, of length numExamples.
    * @param predicted Predicted label vector, of length numExamples.
    * @param paramsList Model parameters of the form List((w, b)), used for calculating regularization cost,
    *                   where w is of shape (d(L-1), d(L)) and b of (d(L)) for Lth layer
    *                   where d(L) is the number of hidden units in Lth layer.
    * @param regularizer Used for calculating regularization cost.
    *                    Potential candidates: VoidRegularizer, new L1Regularizer, new L2Regularizer
    * @return value of the cost function.
    */
  private def calCost(label: DenseVector[Double], predicted: DenseVector[Double],
                      paramsList: NNParams,
                      regularizer: Regularizer): Double = {

    val originalCost = -(label.t * log(predicted + 1E-9) + (1.0 - label).t * log(1.0 - predicted + 1E-9)) / label.length.toDouble
    val reguCost = regularizer.getReguCost(paramsList)

    originalCost + regularizer.lambda * reguCost / label.length.toDouble
  }

  /**
    * Backward propagation of all layers.
    * @param label True label vector of length numExamples.
    * @param forwardResList List of ForwardRes, consisting of yPrevious, zCurrent, yCurrent.
    * @param paramsList List of model paramaters of the form List((w, b)).
    * @param regularizer Used for updating the gradients.
    * @return List of BackwardRes, consisting of (dWCurrent, dBCurrent, dYPrevious),
    *         where zCurrent = yPrevious * wCurrent + OneColumnVector * bCurrent.t,
    *         yCurrent = g(zCurrent), where g is the activation function of the current layer.
    *         So, dZCurrent = dYCurrent *:* g'(zCurrent), where *:* is element-wise multiplication,
    *         dWCurrent = yPrevious.t * dZCurrent / numExamples + lambda * wCurrent / numExamples (due to regulation),
    *         dBCurrent = dZCurrent.t * OneColumnVector / numExamples,
    *         dYPrevious = dZCurrent * wCurrent.t
    */
  private def backward(label: DenseVector[Double],
                       forwardResList: List[ResultUtils.ForwardRes],
                       paramsList: NNParams,
                       regularizer: Regularizer): List[BackwardRes] = {

    val yPredicted = forwardResList.last.yCurrent(::, 0)
    val numExamples = label.length

    val dYPredicted = -(label /:/ (yPredicted + 1E-9) - (1.0 - label) /:/ (1.0 - yPredicted + 1E-9)) // gradients of cross-entropy function.

    //dYHat should be a matrix in the following calculation, but it's just a vector. So convert a vector to a (n, 1) matrix
    val dYHat = DenseMatrix.zeros[Double](numExamples, 1)
    dYHat(::, 0) := dYPredicted

    val initBackwardRes = BackwardRes(dYHat, null, null) // constructing a BackwardRes for cost function.

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
          case _: DropoutLayer => backwardRes // for DropoutLayer, parameters are all ONEs, never need to be updated.
          case _ =>
            new BackwardRes(backwardRes.dYPrevious,
              backwardRes.dWCurrent + regularizer.getReguCostGrad(w, numExamples),
              backwardRes.dBCurrent)
        }
    }
      .dropRight(1)
  }

  /**
    *Train the model using Adam Optimization method. Recommended to use.
    * @param feature
    * @param label
    * @param op MUST be AdamOptimizer
    * @return Updated list of model parameters, of the form List((w, b)).
    */
  private def trainByAdam(feature: DenseMatrix[Double],
                                       label: DenseVector[Double],
                                       op: AdamOptimizer): NNParams = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    val initModelParams = this.weightsInitializer.init(inputDim, this.allLayers)
    val initMomentum = op.initMomentumOrAdam(inputDim, this.allLayers)
    val initAdam = op.initMomentumOrAdam(inputDim, this.allLayers)
    val initParams = AdamOptimizationParams(initModelParams, initMomentum, initAdam) //Combine model parameters, momentum and adam into one case class. Convenient for the following foldLeft operation.

    //TODO refactor mini-batches datasets to Iterator, NOT a List, for saving the storage memory.
    val iterationWithMiniBatches = getIterationWithMiniBatches(feature, label, this.iterationTime, op)

    val trainedParams = iterationWithMiniBatches
      .foldLeft[AdamOptimizationParams](initParams){(previousAdamParams, batchData) =>
      val iteration = batchData._1
      val batchFeature = batchData._2
      val batchLabel = batchData._3
      val miniBatchTime = batchData._4

      val forwardResList = forward(batchFeature, previousAdamParams.modelParams)
      val cost = calCost(batchLabel, forwardResList.last.yCurrent(::, 0),
        previousAdamParams.modelParams, this.regularizer)

      val printMiniBatchUnit = ((numExamples / op.getMiniBatchSize).toInt / 5).toInt //for each iteration, only print minibatch cost FIVE times.
      if (miniBatchTime % printMiniBatchUnit == 0)
        logger.info("Iteration: " + iteration + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
      costHistory.+=(cost)

      val backwardResList = backward(batchLabel, forwardResList, previousAdamParams.modelParams, this.regularizer)
      op.updateParams(previousAdamParams.modelParams, previousAdamParams.momentumParams, previousAdamParams.adamParams, this.learningRate, backwardResList, iteration, miniBatchTime, this.allLayers)
    }.modelParams

    val forwardRes = forward(feature, trainedParams)
    val totalCost = calCost(label, forwardRes.last.yCurrent(::, 0), trainedParams, this.regularizer) //Cost on the entire training set.
    logger.info("Cost on the entire training set: " + totalCost)

    trainedParams
  }

  /**
    * Train without momentum or adam. Maybe Batch gradient descent or Mini-batch gradient descent.
    * @param feature
    * @param label
    * @param op MUST be Non-heuristic optimizer. Maybe GDOptimizer or new SGDOptimizer.
    * @return Updated list of model parameters, of the form List((w, b)).
    */
  private def trainWithoutMomentum(feature: DenseMatrix[Double],
                                   label: DenseVector[Double],
                                   op: NonHeuristic): NNParams = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    val initParams = this.weightsInitializer.init(inputDim, this.allLayers)

    //TODO refactor mini-batches data sets to Iterator, NOT a List, for saving storage memory.
    val iterationWithMiniBatches = op match {
      case op: MiniBatchable => getIterationWithMiniBatches(feature, label, this.iterationTime, op)
      case _ => getIterationWithOneBatch(feature, label, this.iterationTime)
    }

    val trainedParams = iterationWithMiniBatches
      .foldLeft[NNParams](initParams){(previousParams, batchIteration) =>

      val iteration = batchIteration._1
      val batchFeature = batchIteration._2
      val batchLabel = batchIteration._3
      val miniBatchTime = batchIteration._4

      val forwardResList = forward(batchFeature, previousParams)
      val cost = calCost(batchLabel, forwardResList.last.yCurrent(::, 0), previousParams, this.regularizer)

      if (miniBatchTime % 10 == 0)
        logger.info("Iteration: " + iteration + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
      costHistory.+=(cost)

      val backwardResList = backward(batchLabel, forwardResList, previousParams, this.regularizer)
      op.updateParams(previousParams, this.learningRate, backwardResList, iteration, this.allLayers)
    }

    val forwardRes = forward(feature, trainedParams)
    val totalCost = calCost(label, forwardRes.last.yCurrent(::, 0), trainedParams, this.regularizer) //Cost on the entire training set.
    logger.info("Cost on the entire training set: " + totalCost)

    trainedParams
  }

  private def getIterationWithMiniBatches(feature: DenseMatrix[Double],
                                          label: DenseVector[Double],
                                          iterationTime: Int,
                                          op: MiniBatchable): Iterator[(Int, DenseMatrix[Double], DenseVector[Double], Int)] = {

    (0 until iterationTime)
      .toIterator
      .map{iteration =>
        val minibatches = op
          .getMiniBatches(feature, label)
          .zipWithIndex

        minibatches
          .map{minibatch =>
            val miniBatchFeature = minibatch._1._1
            val miniBatchLabel = minibatch._1._2
            val miniBatchTime = minibatch._2
            (iteration, miniBatchFeature, miniBatchLabel, miniBatchTime)
          }
      }
      .flatten
  }

  private def getIterationWithOneBatch(feature: DenseMatrix[Double], label: DenseVector[Double], iterTimes: Int): Iterator[(Int, DenseMatrix[Double], DenseVector[Double], Int)] = {
    (0 until iterTimes)
      .toIterator
      .map{iter =>
        (iter, feature, label, 0)
      }
  }

}

object NeuralNetworkModel {
  def apply(hiddenLayers: List[Layer], outputLayer: Layer,
           learningRate: Double = 0.001, iterTimes: Int = 300,
           weightsInitializer: WeightsInitializer = HeInitializer,
           regularizer: Regularizer = VoidRegularizer,
           optimizer: Optimizer = AdamOptimizer()): NeuralNetworkModel = {
    new NeuralNetworkModel()
      .setHiddenLayerStructure(hiddenLayers)
      .setOutputLayerStructure(outputLayer)
      .setWeightsInitializer(weightsInitializer)
      .setRegularizer(regularizer)
      .setOptimizer(optimizer)
      .setLearningRate(learningRate)
      .setIterationTime(iterTimes)
  }
}
