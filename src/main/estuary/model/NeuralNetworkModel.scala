package estuary.model

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.log
import estuary.components.initializer.{HeInitializer, WeightsInitializer}
import estuary.components.layers.{DropoutLayer, EmptyLayer, Layer}
import estuary.components.optimizer.{AdamOptimizer, Optimizer}
import estuary.components.regularizer.{Regularizer, VoidRegularizer}
import org.apache.log4j.Logger

import scala.collection.mutable


/**
  * Created by meng.pan on 2017/9/5.
  */
class NeuralNetworkModel extends Model {
  override val logger: Logger = Logger.getLogger(this.getClass)

  //Hyperparameters
  protected var hiddenLayers: List[Layer] = _
  protected var outputLayer: Layer = _
  protected var weightsInitializer: WeightsInitializer = HeInitializer //initialization methods, see also HeInitializer, XaiverInitializer
  protected var regularizer: Regularizer = VoidRegularizer //VoidRegularizer: No regularization. see also L1Regularizer, L2Regularizer
  protected var optimizer: Optimizer = AdamOptimizer() //AdamOptimizer, see also GDOptimizer: Batch Gradient Descent, SGDOptimizer,

  private var labelsMapping: List[Double] = _

  def setHiddenLayerStructure(hiddenLayers: Layer*): this.type = {
    this.hiddenLayers = hiddenLayers.toList
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


  private var feature: DenseMatrix[Double] = _
  private var label: DenseVector[Double] = _

  def adddata(feature: DenseMatrix[Double], label: DenseVector[Double]): this.type = {
    this.feature = feature
    this.label = label
    this
  }

  /**
    * Train the model.
    * If the optimizer chosen is AdamOptimizer, then use method trainByAdam.
    * else use trainWithoutMomentum.
    *
    * @param feature Feature matrix whose shape is (numTrainingExamples, inputFeatureDim)
    * @param label   Label vector whose length is numTrainingExamples
    * @todo when any other optimizer is created, the case MUST be added here
    * @return A trained model whose model parameters have been updated
    */
  override def train(feature: DenseMatrix[Double], label: DenseVector[Double]): this.type = {

    initNN()

    val allLayers = hiddenLayers ::: outputLayer :: Nil //concat hidden layers and output layer
    val labelMatrix = Model.convertVectorToMatrix(label)
    this.labelsMapping = label.toArray.toSet.toList.sorted

    val initParams = {
      allLayers.head.setPreviousHiddenUnits(feature.cols)
      allLayers.last.setNumHiddenUnits(this.labelsMapping.length)
      allLayers.par.map(_.init(weightsInitializer)).toList
    }

    val regularizer: Option[Regularizer] = this.regularizer match {
      case null => None
      case VoidRegularizer => None
      case _ => Some(this.regularizer)
    }

    def setLayerParams = (params: List[DenseMatrix[Double]]) => allLayers.zip(params).par.foreach {
      case (layer, param) =>
        layer.setParam(param)
    }

    val forwardAction = (feature: DenseMatrix[Double], label: DenseMatrix[Double], params: List[DenseMatrix[Double]]) => {
      setLayerParams(params)
      val yHat = forward(allLayers, feature)
      calCost(label, yHat, allLayers, this.regularizer)
    }

    val trainedParams = this.optimizer.optimize(feature, labelMatrix)(initParams)(forwardAction) {
      backward(allLayers, regularizer)
    }

    setLayerParams(trainedParams)

    val yHat = forward(allLayers, feature)
    val cost = calCost(labelMatrix, yHat, allLayers, this.regularizer)

    logger.info("Cost on the entire training set: " + cost)

    this
  }

  /**
    * Do predictions. Note: prediction is done without the dropout layer.
    *
    * @param feature Feature matrix with shape (numExamples, inputFeatureDim)
    * @return Predicted labels
    */
  override def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {
    val allLayers = hiddenLayers ::: outputLayer :: Nil //concat hidden layers and output layer

    val nonDropoutLayers = allLayers.par.filter(!_.isInstanceOf[DropoutLayer])
    val predicted = nonDropoutLayers.foldLeft(feature) {
      case (yPrevious, layer) =>
        layer.forwardForPrediction(yPrevious)
    }

    val predictedMatrix = DenseMatrix.zeros[Double](feature.rows, predicted.cols)
    for (i <- (0 until predicted.rows).par) {
      val sliced = predicted(i, ::)
      val maxRow = max(sliced)
      predictedMatrix(i, ::) := sliced.t.map(index => if (index == maxRow) 1.0 else 0.0).t
    }

    val predictedVector = Model.convertMatrixToVector(predictedMatrix, this.labelsMapping)
    predictedVector
  }

  private def initNN(): Unit = {

    initLayerStructure()
    initOptimizer()
  }

  private def initLayerStructure(): Unit = {

    //if a layer is Dropout Layer，then set its number of hidden units to be same as its previous layer
    hiddenLayers = hiddenLayers.scanLeft[Layer, Seq[Layer]](EmptyLayer) {
      (previousLayer, currentLayer) =>
        if (currentLayer.isInstanceOf[DropoutLayer])
          currentLayer.setNumHiddenUnits(previousLayer.numHiddenUnits).setPreviousHiddenUnits(previousLayer.numHiddenUnits)
        else currentLayer.setPreviousHiddenUnits(previousLayer.numHiddenUnits)
    }.toList.tail

    this.outputLayer.setPreviousHiddenUnits(this.hiddenLayers.last.numHiddenUnits)
  }

  private def initOptimizer(): Unit = {
    this.optimizer.setIteration(iterationTime).setLearningRate(learningRate)
  }

  /**
    * Forward propagation of all layers.
    *
    * @param feature Feature matrix with shape (numExamples, inputFeatureDim)
    * @return List of ForwardRes consisting of (yPrevious, zCurrent, yCurrent)
    *         where yPrevious: the output from previous layer, say L-1 th layer, of shape (numExamples, d(L-1))
    *         zCurrent = yPrevious * w(L) + DenseVector.ones[Double](numExamples) * b(L).t
    *         yCurrent = g(zCurrent) where g is the activation function in Lth layer.
    */
  private def forward(layers: Seq[Layer], feature: DenseMatrix[Double]): DenseMatrix[Double] = {
    layers.foldLeft[DenseMatrix[Double]](feature) { case (yPrevious, layer) =>
      layer.forward(yPrevious)
    }
  }

  /**
    * Calculate value of the cost function, consisting the sum of cross-entropy cost and regularization cost.
    *
    * @param label       True transformed label matrix, of shape (numExamples, numOutputUnits)
    * @param predicted   Predicted label matrix, of shape (numExamples, numOutputUnits)
    * @param regularizer Used for calculating regularization cost.
    *                    Potential candidates: VoidRegularizer, new L1Regularizer, new L2Regularizer
    * @return value of the cost function.
    */
  private def calCost(label: DenseMatrix[Double], predicted: DenseMatrix[Double],
                      layers: Seq[Layer],
                      regularizer: Regularizer): Double = {
    val originalCost = -sum(label *:* log(predicted + 1E-9)) / label.rows.toDouble
    val reguCost = layers.foldLeft[Double](0.0) { case (totalReguCost, layer) =>
      totalReguCost + layer.getReguCost(regularizer)
    }

    originalCost + regularizer.lambda * reguCost / label.rows.toDouble
  }

  private def backward(allLayers: List[Layer], regularizer: Option[Regularizer])(label: DenseMatrix[Double], params: List[DenseMatrix[Double]]): List[DenseMatrix[Double]] = {
    allLayers.scanRight((label, DenseMatrix.zeros[Double](1, 1))) { case (layer, (dYCurrent, _)) =>
      layer.backward(dYCurrent, regularizer)
    }.init.map(_._2).seq.toList
  }

  override def getCostHistory: mutable.MutableList[Double] = optimizer.costHistory

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
