package estuary.demo

import estuary.components.layers.{ReluLayer, SoftmaxLayer}
import estuary.components.optimizer.DecentralizedAdamAkkaParallelOptimizer
import estuary.data.GasCensorDataReader
import estuary.model.{FullyConnectedNNModel, Model}

/**
  * Created by mengpan on 2017/10/27.
  */
object AkkaParallelDemo extends App{
  val hiddenLayers = List(
    ReluLayer(numHiddenUnits = 128),
    ReluLayer(numHiddenUnits = 64))
  val outputLayer = SoftmaxLayer()
  val nnModel = new FullyConnectedNNModel(hiddenLayers, outputLayer, None)
  val trainedModel = nnModel.multiNodesParTrain(DecentralizedAdamAkkaParallelOptimizer())

  val (testFeature, testLabel) = new GasCensorDataReader().read("/Users/mengpan/Downloads/NewDataset/test.*")
  val yPredicted = trainedModel.predictToVector(testFeature, Vector(1, 2, 3, 4, 5, 6))
  val testAccuracy = Model.accuracy(Model.convertMatrixToVector(testLabel.map(_.toInt), Vector(1, 2, 3, 4, 5, 6)), yPredicted)
  println("\n The test accuracy of this model is: " + testAccuracy)

  trainedModel.plotCostHistory()
}
