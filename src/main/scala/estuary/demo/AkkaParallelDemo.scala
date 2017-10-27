package estuary.demo

import estuary.components.layers.{ReluLayer, SoftmaxLayer}
import estuary.components.optimizer.AdamAkkaParallelOptimizer
import estuary.model.FullyConnectedNNModel

/**
  * Created by mengpan on 2017/10/27.
  */
object AkkaParallelDemo extends App{
  val hiddenLayers = List(
    ReluLayer(numHiddenUnits = 128),
    ReluLayer(numHiddenUnits = 64))
  val outputLayer = SoftmaxLayer()
  val nnModel = new FullyConnectedNNModel(hiddenLayers, outputLayer, None)
  val trainedModel = nnModel.multiNodesParTrain(AdamAkkaParallelOptimizer())
}
