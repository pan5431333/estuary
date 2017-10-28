package estuary.components.optimizer

import estuary.model.Model

/**
  *
  * @tparam M type of model parameters
  */
trait AkkaParallelOptimizer[M <: AnyRef] extends Optimizer with Serializable {

  /**
    * Optimize the model in parallel, and returning the trained parameters with the same dimensions of initParams.
    * The method parameter 'model' is used here to create several model instances (with copyStructure() method), and
    * then they are distributed to different threads or machines.
    *
    * @param model      an instance of trait Model, used to create many copies and then distribute them to different threads
    *                   or machines.
    * @param initParams initial parameters.
    * @return trained parameters, with same dimension with the given initial parameters.
    */
  def parOptimize(model: Model[M]): M
}


