package estuary.components.optimizer

/**
  * All distributed optimizer MUST implement this trait.
  * @tparam A type of optimization algorithm's parameters.
  */
trait AbstractDistributed[A, B] extends Distributed[B] {

  protected var parameterServer: A

  protected def updateParameterServer(grads: A, miniBatchTime: Int): Unit

  protected def fetchParameterServer(): A

}
