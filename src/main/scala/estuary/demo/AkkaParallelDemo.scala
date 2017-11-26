package estuary.demo

import estuary.components.layers.ConvLayer.{ConvSize, Filter}
import estuary.components.layers._
import estuary.components.optimizer.{AdamAkkaParallelOptimizer, AdamOptimizer, DecentralizedAdamAkkaParallelOptimizer}
import estuary.data.GasCensorDataReader
import estuary.model.{FullyConnectedNNModel, Model}

/**
  * Created by mengpan on 2017/10/27.
  */
object AkkaParallelDemo extends App{
//  val hiddenLayers = List(
//    ReluConvLayer(Filter(size = 2, pad = 0, stride = 1, oldChannel = 3, newChannel = 8), preConvSize = ConvSize(32, 32, 3)),
//    ReluConvLayer(Filter(size = 3, pad = 0, stride = 1, oldChannel = 8, newChannel = 16), preConvSize = ConvSize(31, 31, 8)),
//    PoolingLayer(3, 1, 0, PoolingLayer.MAX_POOL, preConvSize = ConvSize(29, 29, 16)))
//  val outputLayer = SoftmaxLayer(2)

  val hiddenLayers = List(
    ReluConvLayer(Filter(size = 2, pad = 0, stride = 1, oldChannel = 2, newChannel = 8), preConvSize = ConvSize(8, 8, 2)),
    ReluConvLayer(Filter(size = 3, pad = 0, stride = 1, oldChannel = 8, newChannel = 16), preConvSize = ConvSize(7, 7, 8)),
    PoolingLayer(2, 1, 0, PoolingLayer.MAX_POOL, preConvSize = ConvSize(5, 5, 16)),
    ReluConvLayer(Filter(size = 2, pad = 0, stride = 1, oldChannel = 16, newChannel = 32), preConvSize = ConvSize(4, 4, 16)),
    ReluConvLayer(Filter(size = 2, pad = 0, stride = 1, oldChannel = 32, newChannel = 64), preConvSize = ConvSize(3, 3, 32)),
    PoolingLayer(2, 1, 0, PoolingLayer.MAX_POOL, preConvSize = ConvSize(2, 2, 64)),
    ReluLayer(32))
  val outputLayer = SoftmaxLayer(6)

  val nnModel = new FullyConnectedNNModel(hiddenLayers, outputLayer, None)

//  val (feature, label) = new GasCensorDataReader().read("/Users/mengpan/Downloads/NewDataset/training.*")

  val trainedModel = nnModel.multiNodesParTrain(AdamAkkaParallelOptimizer(miniBatchSize = 64, learningRate = 0.001))
//  val trainedModel = nnModel.train(feature, label, AdamOptimizer())

  val (testFeature, testLabel) = new GasCensorDataReader().read("""D:\\Users\\m_pan\\Downloads\\Dataset\\Dataset\\test.*""")
  val yPredicted = trainedModel.predictToVector(testFeature, Vector(1, 2, 3, 4, 5, 6))
  val testAccuracy = Model.accuracy(Model.convertMatrixToVector(testLabel.map(_.toInt), Vector(1, 2, 3, 4, 5, 6)), yPredicted)
  println("\n The test accuracy of this model is: " + testAccuracy)

  trainedModel.plotCostHistory()
}
