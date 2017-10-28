package estuary.components.optimizer

import breeze.linalg.DenseMatrix
import estuary.components.optimizer.AdamOptimizer.AdamParam

/**
  * Created by mengpan on 2017/10/28.
  */
class DecentralizedAdamAkkaParallelOptimizer(override val iteration: Int,
                                             override val learningRate: Double,
                                             override val paramSavePath: String,
                                             override val miniBatchSize: Int,
                                             override val momentumRate: Double,
                                             override val adamRate: Double)
  extends AdamOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate)
    with DecentralizedAkkaParallelOptimizer[AdamParam, Seq[DenseMatrix[Double]]] {
  override protected def avgOpFunc(params: Seq[AdamParam]): AdamParam = {
    val n = params.length.toDouble
    val sum = params.reduce[AdamParam] { case (a1, a2) =>
        AdamParam(addParams(a1.modelParam, a2.modelParam), addParams(a1.momentumParam, a2.momentumParam), addParams(a1.adamParam, a2.adamParam))
    }
    AdamParam(sum.modelParam.map(_/n), sum.momentumParam.map(_/n), sum.adamParam.map(_/n))
  }

  protected def addParams(a: Seq[DenseMatrix[Double]], b: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    a.zip(b).map{ case (am, bm) =>
        am + bm
    }
  }

  /**
    * Given model parameters to initialize optimization parameters, i.e. for Adam Optimization, model parameters are of type
    * "Seq[DenseMatrix[Double] ]", optimization parameters are of type "AdamParams", i.e. case class of
    * (Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ])
    *
    * @param modelParams model parameters
    * @return optimization parameters
    */
  override protected def modelParamsToOpParams(modelParams: Seq[DenseMatrix[Double]]): AdamParam = AdamParam(modelParams, getInitAdam(modelParams), getInitAdam(modelParams))

  override protected def opParamsToModelParams(opParams: AdamParam): Seq[DenseMatrix[Double]] = opParams.modelParam
}

object DecentralizedAdamAkkaParallelOptimizer {
  def apply(iteration: Int = 100, learningRate: Double = 0.001, paramSavePath: String = System.getProperty("user.dir"), miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999): DecentralizedAdamAkkaParallelOptimizer = {
    new DecentralizedAdamAkkaParallelOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate)
  }
}
