package estuary.components.optimizer

/**
  * All distributed optimizer MUST implement this trait.
  * @tparam T type of optimization algorithm's parameters.
  */
trait AbstractDistributed[T] extends Distributed {

  protected var parameterServer: T

  protected def updateParameterServer(grads: T, miniBatchTime: Int): Unit

  protected def fetchParameterServer(): T

}
