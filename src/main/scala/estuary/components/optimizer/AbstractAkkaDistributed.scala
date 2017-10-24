package estuary.components.optimizer

import java.util.concurrent.TimeUnit

import akka.actor.{Actor, ActorSystem, AddressFromURIString, Deploy, Props, Terminated}
import akka.remote.RemoteScope
import breeze.linalg.DenseMatrix
import com.typesafe.config.ConfigFactory
import estuary.components.optimizer.AbstractAkkaDistributed.{CostHistory, Start}
import estuary.concurrency.BatchGradCalculatorActor.StartTrain
import estuary.concurrency.ParameterServerActor.{CurrentParams, GetCurrentParams}
import estuary.concurrency.{BatchGradCalculatorActor, ParameterServerActor}
import estuary.model.Model

/**
  *
  * @tparam O type of optimization algorithm's parameters.
  * @tparam M type of model parameters
  */
trait AbstractAkkaDistributed[O, M] extends AbstractDistributed[ParameterServerActor[O], M] with Serializable{

  protected var nowParams: O = _

  final override protected def updateParameterServer(grads: ParameterServerActor[O], miniBatchTime: Int): Unit = {}

  /**
    * Given model parameters to initialize optimization parameters, i.e. for Adam Optimization, model parameters are of type
    * "Seq[DenseMatrix[Double] ]", optimization parameters are of type "AdamParams", i.e. case class of
    * (Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ], Seq[DenseMatrix[Double] ])
    * @param modelParams model parameters
    * @return optimization parameters
    */
  protected def modelParamsToOpParams(modelParams: M): O

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

    val config = ConfigFactory.load("estuary")

    //Create parameter server actor
    val parameterServerAddress = AddressFromURIString(config.getString("estuary.parameter-server"))
    val init = modelParamsToOpParams(initParams)
    val paramServerActor = AbstractAkkaDistributed.system.actorOf(Props(
      new ParameterServerActor[O](init)).withDeploy(Deploy(scope = RemoteScope(parameterServerAddress))), name="parameterServerActor")

    //Create work actors (for calculating cost and grads)
    val workersAddress = config.getStringList("estuary.workers")
    val nWorkers = workersAddress.size()
    val nTasksPerWorker = math.ceil(nTasks / nWorkers.toDouble).toInt
    val workActors = batches.zip(models).zipWithIndex.map { case ((batch, eModel), taskIndex) =>
        val workerIndex = taskIndex / nTasksPerWorker
        AbstractAkkaDistributed.system.actorOf(Props(new BatchGradCalculatorActor[M, O](
          batch._1, batch._2, eModel, iteration, getMiniBatches, updateFunc, paramServerActor, opParamsToModelParams
        )).withDeploy(Deploy(scope = RemoteScope(AddressFromURIString(workersAddress.get(workerIndex))))))
      }

    //Responsible for making all Workers to start, watching if training finished, getting the trained parameters from
    //parameterServer actor after training finished, and storing cost history from each worker during training.
    val mainActor = AbstractAkkaDistributed.system.actorOf(Props(new Actor{
      private var nWorksDead: Int = 0

      override def receive: Actor.Receive = {

        case Start =>
          //Start every work actors
          workActors foreach { a => a ! StartTrain; TimeUnit.SECONDS.sleep(1)}
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

    //Shutdown main actor, however, parameterServer actor and all worker actors are not under control of this actor system,
    //hence they will not be terminated.
    AbstractAkkaDistributed.system.terminate()

    opParamsToModelParams(nowParams)
  }
}

object AbstractAkkaDistributed {
  val system = ActorSystem("MainSystem", ConfigFactory.load("MainActorSystem"))

  sealed trait AbstractAkkaDistributedMsg
  final case class CostHistory(cost: Double) extends AbstractAkkaDistributedMsg
  final case object Start extends AbstractAkkaDistributedMsg
}
