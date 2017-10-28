package estuary.components.optimizer

import breeze.linalg.DenseMatrix

class SGDAkkaParallelOptimizer(override val iteration: Int,
                                override val learningRate: Double,
                                override val paramSavePath: String,
                                override val miniBatchSize: Int)
  extends SGDOptimizer(iteration, learningRate, paramSavePath, miniBatchSize)
    with CentralizedAkkaParallelOptimizer[Seq[DenseMatrix[Double]], Seq[DenseMatrix[Double]]] {

  /**
    * Given model parameters to initialize optimization parameters, i.e. for Adam Optimization, model parameters are of type
    * "Seq[DenseMatrix[Double] ]", optimization parameters are of type "AdamParams", i.e. case class of
    * (Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ])
    *
    * @param modelParams model parameters
    * @return optimization parameters
    */
  protected def modelParamsToOpParams(modelParams: Seq[DenseMatrix[Double]]) = modelParams

  protected def opParamsToModelParams(opParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = opParams
}

object SGDAkkaParallelOptimizer {
  def apply(iteration: Int = 100, learningRate: Double = 0.001, paramSavePath: String = System.getProperty("user.dir"), miniBatchSize: Int = 64): SGDAkkaParallelOptimizer = {
    new SGDAkkaParallelOptimizer(iteration, learningRate, paramSavePath, miniBatchSize)
  }
}
