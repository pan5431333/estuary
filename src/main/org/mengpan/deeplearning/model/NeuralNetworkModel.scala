package org.mengpan.deeplearning.model
import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{log, pow}
import org.apache.log4j.Logger
import org.mengpan.deeplearning.components.initializer.{HeInitializer, NormalInitializer, WeightsInitializer, XaiverInitializer}
import org.mengpan.deeplearning.components.layers.{DropoutLayer, EmptyLayer, Layer}
import org.mengpan.deeplearning.components.optimizer._
import org.mengpan.deeplearning.components.regularizer.{L1Regularizer, L2Regularizer, Regularizer, VoidRegularizer}


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
        currentLayer.setNumHiddenUnits(previousLayer.numHiddenUnits).setPreviousHiddenUnits(previousLayer.numHiddenUnits)
      else currentLayer.setPreviousHiddenUnits(previousLayer.numHiddenUnits)
    }.toList

    this.hiddenLayers = theHiddenLayer.tail //drop the first EmptyLayer in the list
    this
  }

  def setOutputLayerStructure(outputLayer: Layer): this.type = {
    this.outputLayer = outputLayer
    this.outputLayer.setPreviousHiddenUnits(this.hiddenLayers.last.numHiddenUnits)
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
    * If the optimizer chosen is AdamOptimizer, then use method trainByAdam.
    * else use trainWithoutMomentum.
    * @param feature Feature matrix whose shape is (numTrainingExamples, inputFeatureDim)
    * @param label Label vector whose length is numTrainingExamples
    * @todo when any other optimizer is created, the case MUST be added here
    * @return A trained model whose model parameters have been updated
    */
  override def train(feature: DenseMatrix[Double], label: DenseVector[Double]): this.type = {
    this.allLayers.head.setPreviousHiddenUnits(feature.cols)
    this.optimizer.setIteration(iterationTime).setLearningRate(learningRate)
    val labelMatrix = convertVectorToMatrix(label)
    val initParams = this.allLayers.map(_.init(weightsInitializer))

    val trainedParams = this.optimizer.optimize(feature, labelMatrix)(initParams){
      case (feature, label, params) =>
            val yHat = forward(this.allLayers, feature)
            calCost(label, yHat, this.allLayers, this.regularizer)
    }{backward}

    this.allLayers.zip(trainedParams).foreach{
      case (layer, param) =>
        layer.setParam(param)
    }

    this
  }

  /**
    * Do predictions. Note prediction is done without the dropout layer.
 *
    * @param feature Feature matrix with shape (numExamples, inputFeatureDim)
    * @return Predicted labels
    */
  override def predict(feature: DenseMatrix[Double]): DenseVector[Double] = {

    val nonDropoutLayers = this.allLayers.filter(!_.isInstanceOf[DropoutLayer])
    val predicted =  forward(nonDropoutLayers, feature)

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
  private def forward(layers: Seq[Layer], feature: DenseMatrix[Double]): DenseMatrix[Double] = {

    layers.foldLeft[DenseMatrix[Double]](feature){
      case (yPrevious, layer) =>
        layer.forward(yPrevious)
    }
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
                      layers: Seq[Layer],
                      regularizer: Regularizer): Double = {
    val originalCost = - sum(label *:* log(predicted + 1E-9)) / label.rows.toDouble
    val reguCost = layers.foldLeft[Double](0.0){
      case (totalReguCost, layer) =>
        totalReguCost + layer.getReguCost(regularizer)
    }

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
  private def backward(label: DenseMatrix[Double], params: List[DenseMatrix[Double]]): List[DenseMatrix[Double]] = {

    this.allLayers.zip(params).foreach{
      case (layer, param) =>
        layer.setParam(param)
    }

    this.allLayers.scanRight((label, DenseMatrix.zeros[Double](1, 1))){
      case (layer, (dYCurrent, _)) =>
        layer.backward(dYCurrent)
    }.init.map(_._2)
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

  override def getCostHistory = optimizer.costHistory

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
