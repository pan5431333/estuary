package estuary.components.optimizer

import estuary.model.Model

/**
  *
  */
trait AkkaParallelOptimizer extends Optimizer with Serializable {

  /**
    * Optimize the model in parallel, and returning the trained parameters with the same dimensions of initParams.
    * The method parameter 'model' is used here to create several model instances (with copyStructure() method), and
    * then they are distributed to different threads or machines.
    *
    * @param model      an instance of trait ModelLike, used to create many copies and then distribute them to different threads
    *                   or machines.
    * @return trained parameters, with same dimension with the given initial parameters.
    */
  def parOptimize[ModelParam](model: Model): ModelParam
}


