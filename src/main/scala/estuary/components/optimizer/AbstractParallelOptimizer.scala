package estuary.components.optimizer

/**
  * All distributed optimizer MUST implement this trait.
 *
  * @tparam OptParam type of optimization algorithm's parameters.
  */
trait AbstractParallelOptimizer[OptParam, ModelParam] extends ParallelOptimizer {

  protected var parameterServer: OptParam = _

  protected def updateParameterServer(grads: OptParam, miniBatchTime: Int): Unit

  protected def updateParameterServer(newParam: OptParam): Unit = this.synchronized(parameterServer = newParam)

  protected def fetchParameterServer(): OptParam = this.synchronized(parameterServer)

}
