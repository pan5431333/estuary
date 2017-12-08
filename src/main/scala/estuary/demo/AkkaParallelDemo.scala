package estuary.demo

import estuary.components.layers.ConvLayer.{ConvSize, Filter}
import estuary.components.layers._
import estuary.components.optimizer._
import estuary.data.GasCensorDataReader
import estuary.model.Model
import shapeless.{HList, HNil}

/**
  * Created by mengpan on 2017/10/27.
  */
object AkkaParallelDemo extends App{
//  val hiddenLayers = List(
//    ReluConvLayer(Filter(size = 2, pad = 0, stride = 1, oldChannel = 3, newChannel = 8), preConvSize = ConvSize(32, 32, 3)),
//    ReluConvLayer(Filter(size = 3, pad = 0, stride = 1, oldChannel = 8, newChannel = 16), preConvSize = ConvSize(31, 31, 8)),
//    PoolingLayer(3, 1, 0, PoolingLayer.MAX_POOL, preConvSize = ConvSize(29, 29, 16)))
//  val outputLayer = SoftmaxLayer(2)

  val hiddenLayers: HList =
    ReluConvLayer(Filter(size = 2, pad = 0, stride = 1, oldChannel = 2, newChannel = 8), preConvSize = ConvSize(8, 8, 2)) ::
    ReluConvLayer(Filter(size = 3, pad = 0, stride = 1, oldChannel = 8, newChannel = 16), preConvSize = ConvSize(7, 7, 8)) ::
    PoolingLayer(2, 1, 0, PoolingLayer.MAX_POOL, preConvSize = ConvSize(5, 5, 16)) :: HNil

  val outputLayer = SoftmaxLayer(6)

  val nnModel = new Model(hiddenLayers, outputLayer)

  val (feature, label) = new GasCensorDataReader().read("""D:\\Users\\m_pan\\Downloads\\Dataset\\Dataset\\train.*""")

//  val trainedModel = nnModel.multiNodesParTrain(AdamAkkaParallelOptimizer(iteration = 10, miniBatchSize = 128, learningRate = 0.005))
  nnModel.train(feature, label, AdamOptimizer(iteration = 20, learningRate = 0.0005))


//  nnModel.plotCostHistory()
}
