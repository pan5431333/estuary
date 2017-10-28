package estuary.components.optimizer

/**
  *
  * @tparam O type of optimization algorithm's parameters.
  * @tparam M type of model parameters
  */
trait AbstractAkkaParallelOptimizer[O <: AnyRef, M <: AnyRef] extends AkkaParallelOptimizer[M] with MiniBatchable with Serializable {

  /**
    * Given model parameters to initialize optimization parameters, i.e. for Adam Optimization, model parameters are of type
    * "Seq[DenseMatrix[Double] ]", optimization parameters are of type "AdamParams", i.e. case class of
    * (Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ])
    *
    * @param modelParams model parameters
    * @return optimization parameters
    */
  protected def modelParamsToOpParams(modelParams: M): O

  protected def updateFunc(opParams: O, grads: M, miniBatchTime: Int): O

  protected def opParamsToModelParams(opParams: O): M

}




