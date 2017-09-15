package org.mengpan.deeplearning.model
import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
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

  private var labelsMapping: List[Double] = _

  def setHiddenLayerStructure(hiddenLayers: Layer*): this.type = {
    if (hiddenLayers.isEmpty) {
      throw new IllegalArgumentException("hidden layer should be at least one layer!")
    }

    //if a layer is Dropout Layer，then set its number of hidden units to be same as its previous layer
    val theHiddenLayer: List[Layer] = hiddenLayers
      .scanLeft[Layer, Seq[Layer]](EmptyLayer){
      (previousLayer, currentLayer) =>
      if (currentLayer.isInstanceOf[DropoutLayer])
        currentLayer.setNumHiddenUnits(previousLayer.numHiddenUnits)
      else currentLayer
    }.toList

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
    val labelMatrix: DenseMatrix[Double] = convertVectorToMatrix(label)
    this.outputLayer.setNumHiddenUnits(this.labelsMapping.size)

    this.paramsList = this.optimizer match {
      case op: AdamOptimizer => trainByAdam(feature, labelMatrix, op)
      case op: NonHeuristic => trainWithoutMomentum(feature, labelMatrix, op)
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
    val predicted = forwardResList.last.yCurrent
    val predictedMatrix = DenseMatrix.zeros[Double](feature.rows, predicted.cols)
    for (i <- 0 until predicted.rows) {
      val sliced = predicted(i, ::)
      val maxRow = max(sliced)
      predictedMatrix(i, ::) := sliced.t.map(d => if (d == maxRow) 1.0 else 0.0).t
    }

    val predictedVector = convertMatrixToVector(predictedMatrix, this.labelsMapping)
    predictedVector
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
    * @param label True transformed label matrix, of shape (numExamples, numOutputUnits)
    * @param predicted Predicted label matrix, of shape (numExamples, numOutputUnits)
    * @param paramsList Model parameters of the form List((w, b)), used for calculating regularization cost,
    *                   where w is of shape (d(L-1), d(L)) and b of (d(L)) for Lth layer
    *                   where d(L) is the number of hidden units in Lth layer.
    * @param regularizer Used for calculating regularization cost.
    *                    Potential candidates: VoidRegularizer, new L1Regularizer, new L2Regularizer
    * @return value of the cost function.
    */
  private def calCost(label: DenseMatrix[Double], predicted: DenseMatrix[Double],
                      paramsList: NNParams,
                      regularizer: Regularizer): Double = {
    val originalCost = - sum(label *:* log(predicted + 1E-9)) / label.rows.toDouble
    val reguCost = regularizer.getReguCost(paramsList)

    originalCost + regularizer.lambda * reguCost / label.rows.toDouble
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
  private def backward(label: DenseMatrix[Double],
                       forwardResList: List[ResultUtils.ForwardRes],
                       paramsList: NNParams,
                       regularizer: Regularizer): List[BackwardRes] = {

    val yPredicted = forwardResList.last.yCurrent
    val numExamples = label.rows

    //HACK!
    val initBackwardRes = BackwardRes(label, null, null)

    paramsList
      .zip(this.allLayers)
      .zip(forwardResList)
      .scanRight[BackwardRes, List[BackwardRes]](initBackwardRes){
      case ((((w, b), layer), forwardRes), previousBackwardRes) =>
        val dYCurrent = previousBackwardRes.dYPrevious

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
                                       label: DenseMatrix[Double],
                                       op: AdamOptimizer): NNParams = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    val initModelParams = this.weightsInitializer.init(inputDim, this.allLayers)
    val initMomentum = op.initMomentumOrAdam(inputDim, this.allLayers)
    val initAdam = op.initMomentumOrAdam(inputDim, this.allLayers)
    val initParams = AdamOptimizationParams(initModelParams, initMomentum, initAdam) //Combine model parameters, momentum and adam into one case class. Convenient for the following foldLeft operation.

    val iterationWithMiniBatches = getIterationWithMiniBatches(feature, label, this.iterationTime, op)

    val trainedParams = iterationWithMiniBatches.foldLeft(initParams){
      case (previousParams, (iteration, batches)) =>
        batches.zipWithIndex.foldLeft[AdamOptimizationParams](previousParams){
          case (previousAdamParams, ((batchFeature, batchLabel), miniBatchTime)) =>
            val forwardResList = forward(batchFeature, previousAdamParams.modelParams)
            val cost = calCost(batchLabel, forwardResList.last.yCurrent,
              previousAdamParams.modelParams, this.regularizer)

            val printMiniBatchUnit = ((numExamples / op.getMiniBatchSize).toInt / 5).toInt //for each iteration, only print minibatch cost FIVE times.
            if (miniBatchTime % printMiniBatchUnit == 0)
              logger.info("Iteration: " + iteration + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
            costHistory.+=(cost)

            val backwardResList = backward(batchLabel, forwardResList, previousAdamParams.modelParams, this.regularizer)
            op.updateParams(previousAdamParams.modelParams, previousAdamParams.momentumParams, previousAdamParams.adamParams, this.learningRate, backwardResList, iteration, miniBatchTime, this.allLayers)
        }
    }.modelParams

    val forwardRes = forward(feature, trainedParams)
    val totalCost = calCost(label, forwardRes.last.yCurrent, trainedParams, this.regularizer) //Cost on the entire training set.
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
                                   label: DenseMatrix[Double],
                                   op: NonHeuristic): NNParams = {
    val numExamples = feature.rows
    val inputDim = feature.cols

    val initParams = this.weightsInitializer.init(inputDim, this.allLayers)

    val iterationWithMiniBatches = op match {
      case op: MiniBatchable => getIterationWithMiniBatches(feature, label, this.iterationTime, op)
      case _ => getIterationWithOneBatch(feature, label, this.iterationTime)
    }

    val trainedParams = iterationWithMiniBatches.foldLeft(initParams){
      case (previousParams, (iteration, batch)) =>
        batch.zipWithIndex.foldLeft(previousParams){
          case (previousBatchParams, ((batchFeature, batchLabel), miniBatchTime)) =>
            val fordwardResList = forward(batchFeature, previousBatchParams)
            val cost = calCost(batchLabel, fordwardResList.last.yCurrent, previousBatchParams, this.regularizer)

            if (miniBatchTime % 10 == 0)
              logger.info("Iteration: " + iteration + "|=" + "=" * (miniBatchTime / 10) + ">> Cost: " + cost)
            costHistory.+=(cost)

            val backwardResList = backward(batchLabel, fordwardResList, previousBatchParams, this.regularizer)
            op.updateParams(previousBatchParams, this.learningRate, backwardResList, iteration, this.allLayers)
        }
    }

    val forwardRes = forward(feature, trainedParams)
    val totalCost = calCost(label, forwardRes.last.yCurrent, trainedParams, this.regularizer) //Cost on the entire training set.
    logger.info("Cost on the entire training set: " + totalCost)

    trainedParams
  }

  private def getIterationWithMiniBatches(feature: DenseMatrix[Double],
                                          label: DenseMatrix[Double],
                                          iterationTime: Int,
                                          op: MiniBatchable): Iterator[(Int, Iterator[(DenseMatrix[Double], DenseMatrix[Double])])] = {

    (0 until iterationTime)
      .toIterator
      .map{iteration =>
        val minibatches = op
          .getMiniBatches(feature, label)

        (iteration, minibatches)
      }

  }

  private def getIterationWithOneBatch(feature: DenseMatrix[Double], label: DenseMatrix[Double], iterTimes: Int): Iterator[(Int, Iterator[(DenseMatrix[Double], DenseMatrix[Double])])] = {
    (0 until iterTimes)
      .toIterator
      .map{iter =>
        (iter, Iterator((feature, label)))
      }
  }

  /**
    * Convert labels in a single vector to a matrix.
    * e.g. Vector(0, 1, 0, 1) => Matrix(Vector(1, 0, 1, 0), Vector(0, 1, 0, 1))
    * Vector(0, 1, 2) => Matrix(Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1))
    * @param labelVector
    * @return
    */
  private def convertVectorToMatrix(labelVector: DenseVector[Double]): DenseMatrix[Double] = {
    val labels = labelVector.toArray.toSet.toList.sorted //distinct elelents by toSet.
    this.labelsMapping = labels

    val numLabels = labels.size
    val res = DenseMatrix.zeros[Double](labelVector.length, numLabels)

    for ((label, i) <- labels.zipWithIndex) {
      val helperVector = DenseVector.ones[Double](labelVector.length) * label
      res(::, i) := elementWiseEqualCompare(labelVector, helperVector)
    }
    res
  }

  private def convertMatrixToVector(labelMatrix: DenseMatrix[Double], labelsMapping: List[Double]): DenseVector[Double] = {
    val labelsMappingVec = labelsMapping.toVector

    val res = DenseVector.zeros[Double](labelMatrix.rows)

    for (i <- 0 until labelMatrix.cols) {
      res :+= labelMatrix(::, i) * labelsMappingVec(i)
    }
    res
  }

  /**
    * Compare two vector for equality in element-wise.
    * e.g. a = Vector(1, 2, 3), b = Vector(1, 0, 0), then return Vector(1, 0, 0)
    * @param a
    * @param b
    * @return
    */
  private def elementWiseEqualCompare(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = {
    assert(a.length == b.length, "a.length != b.length")
    val compareArr = a.toArray.zip(b.toArray).map{case (i, j) =>
      if (i == j) 1.0 else 0.0
    }
    DenseVector(compareArr)
  }

}

object NeuralNetworkModel {
  def apply(hiddenLayers: Seq[Layer], outputLayer: Layer,
           learningRate: Double = 0.001, iterTimes: Int = 300,
           weightsInitializer: WeightsInitializer = HeInitializer,
           regularizer: Regularizer = VoidRegularizer,
           optimizer: Optimizer = AdamOptimizer()): NeuralNetworkModel = {
    new NeuralNetworkModel()
      .setHiddenLayerStructure(hiddenLayers: _*)
      .setOutputLayerStructure(outputLayer)
      .setWeightsInitializer(weightsInitializer)
      .setRegularizer(regularizer)
      .setOptimizer(optimizer)
      .setLearningRate(learningRate)
      .setIterationTime(iterTimes)
  }
}
