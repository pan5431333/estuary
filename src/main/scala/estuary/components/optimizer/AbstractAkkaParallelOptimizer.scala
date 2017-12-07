package estuary.components.optimizer

/**
  *
  * @tparam OptParam type of optimization algorithm's parameters.
  * @tparam ModelParam type of model parameters
  */
trait AbstractAkkaParallelOptimizer[OptParam, ModelParam] extends AkkaParallelOptimizer[ModelParam] with MiniBatchable with Serializable {

  /**
    * Given model parameters to initialize optimization parameters, i.e. for Adam Optimization, model parameters are of type
    * "Seq[DenseMatrix[Double] ]", optimization parameters are of type "AdamParams", i.e. case class of
    * (Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ])
    *
    * @param modelParams model parameters
    * @return optimization parameters
    */
  protected def modelParamsToOpParams(modelParams: ModelParam): OptParam

  protected def updateFunc(opParams: OptParam, grads: ModelParam, miniBatchTime: Int): OptParam

  protected def opParamsToModelParams(opParams: OptParam): ModelParam

}




