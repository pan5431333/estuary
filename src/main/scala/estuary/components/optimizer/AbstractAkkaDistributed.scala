package estuary.components.optimizer

import akka.actor.{Actor, ActorSystem, Props, Terminated}
import breeze.linalg.DenseMatrix
import estuary.components.optimizer.AbstractAkkaDistributed.CostHistory
import estuary.concurrency.BatchGradCalculatorActor.Start
import estuary.concurrency.ParameterServerActor.{CurrentParams, GetCurrentParams}
import estuary.concurrency.{BatchGradCalculatorActor, ParameterServerActor}
import estuary.model.Model

/**
  *
  * @tparam O type of optimization algorithm's parameters.
  * @tparam M type of model parameters
  */
trait AbstractAkkaDistributed[O, M] extends AbstractDistributed[ParameterServerActor[O], M]{

  protected var nowParams: O = _
  protected var isLocal: Boolean = _

  final override protected def updateParameterServer(grads: ParameterServerActor[O], miniBatchTime: Int): Unit = {}

  /**
    * Given model parameters to initialize optimization parameters, i.e. for Adam Optimization, model parameters are of type
    * "Seq[DenseMatrix[Double] ]", optimization parameters are of type "AdamParams", i.e. case class of
    * (Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ])
    * @param modelParams model parameters
    * @return optimization parameters
    */
  protected def initOpParamsByModelParams(modelParams: M): O

  protected def updateFunc(opParams: O, grads: M, miniBatchTime: Int): O

  protected def opParamsToModelParams(opParams: O): M

  /**
    * Optimize the model in parallel, and returning the trained parameters with the same dimensions of initParams.
    * The method parameter 'model' is used here to create several model instances (with copyStructure() method), and
    * then they are distributed to different threads or machines.
    *
    * @param feature    feature matrix
    * @param label      label matrix with one-hot representation.
    * @param model      an instance of trait Model, used to create many copies and then distribute them to different threads
    *                   or machines.
    * @param initParams initial parameters.
    * @return trained parameters, with same dimension with the given initial parameters.
    */
  override def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model[M], initParams: M): M = {
    val batches = genParBatches(feature, label).seq
    val models = batches.indices.map(_ => model.copyStructure)
    @volatile
    var isDone: Boolean = false
    @volatile
    var isDone2: Boolean = false

    val mainActor = AbstractAkkaDistributed.system.actorOf(Props(new Actor{
      //Create parameter server actor
      private val init = initOpParamsByModelParams(initParams)
      private val paramServerActor =
        if (isLocal)
          context.actorOf(Props(new ParameterServerActor[O](init)), name="ParameterServerActor")
        else {
          //todo If parameterServer is on remote, then lookup actor on that remote server
          context.actorOf(Props(new ParameterServerActor[O](init)), name="ParameterServerActor")
        }

      //Create work actors (for calculating cost and grads)
      private val workActors =
        if (isLocal) {
          batches.zip(models).map { case (batch, eModel) => context.actorOf(Props(
            new BatchGradCalculatorActor[M, O](
              batch._1, batch._2, eModel, iteration, getMiniBatches, updateFunc, paramServerActor, opParamsToModelParams)
          ))}
        } else {
          //todo If workerActors are on remote, then lookup actors on those remote servers
          batches.zip(models).map { case (batch, eModel) => context.actorOf(Props(
            new BatchGradCalculatorActor[M, O](
              batch._1, batch._2, eModel, iteration, getMiniBatches, updateFunc, paramServerActor, opParamsToModelParams)
          ))}
        }

      private var nWorksDead: Int = 0

      override def receive: Actor.Receive = {
        case Start =>
          //Start every work actors
          workActors foreach {_ ! Start}
          workActors foreach {a => context.watch(a)}

        case Terminated(_) =>
          nWorksDead += 1
          if (nWorksDead == nTasks) {isDone = true; paramServerActor ! GetCurrentParams}

        case CurrentParams(params) =>
          nowParams = params.asInstanceOf[O]
          isDone2 = true
          context.stop(self)

        case CostHistory(cost) => addCostHistory(cost)
      }
    }))

    //Start training
    mainActor ! Start

    //Waiting for training to be done.
    while (!isDone || !isDone2) {}

    //Shutdown main actor, parameterServer actor and all worker actors
    AbstractAkkaDistributed.system.terminate()

    opParamsToModelParams(nowParams)
  }
}

object AbstractAkkaDistributed {
  val system = ActorSystem("AbstractAkkaDistributed")

  sealed trait AbstractAkkaDistributedMsg
  case class CostHistory(cost: Double) extends AbstractAkkaDistributedMsg
}
