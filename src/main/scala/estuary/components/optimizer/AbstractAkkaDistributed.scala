package estuary.components.optimizer

import java.io.FileNotFoundException

import akka.actor.{ActorSystem, AddressFromURIString, Deploy, Props}
import akka.remote.RemoteScope
import akka.util.Timeout
import breeze.linalg.DenseMatrix
import com.typesafe.config.ConfigFactory
import estuary.concurrency.BatchGradCalculatorActor.StartTrain
import estuary.concurrency.ParameterServerActor.{CurrentParams, GetCostHistory, GetTrainedParams, SetWorkActorsRef}
import estuary.concurrency.{BatchGradCalculatorActor, Manager, ParameterServerActor, createActorSystem}
import estuary.model.Model

import scala.concurrent.{Await, Future}

/**
  *
  * @tparam O type of optimization algorithm's parameters.
  * @tparam M type of model parameters
  */
trait AbstractAkkaDistributed[O, M] extends AbstractDistributed[ParameterServerActor[O], M] with Serializable {

  final override protected def updateParameterServer(grads: ParameterServerActor[O], miniBatchTime: Int): Unit = {}

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

    val config = ConfigFactory.load("estuary")

    val system = try {
      createActorSystem("MainSystem", "MainSystem")
    } catch {
      case _: FileNotFoundException =>
        logger.warn("No configuration file MainSystem found, use default akka system: akka.tcp://MainSystem@127.0.0.1:2552")
        createActorSystem("MainSystem")
    }

    logger.info("MainSystem {} created", system)

    //Create parameter server actor (storing parameters)
    val parameterServerAddress = AddressFromURIString(config.getString("estuary.parameter-server"))
    val init = modelParamsToOpParams(initParams)
    val paramServerActor = system.actorOf(Props(
      new ParameterServerActor[O](init)).withDeploy(Deploy(scope = RemoteScope(parameterServerAddress))), name = "parameterServerActor")

    logger.info(s"Parameter Server Actor ($paramServerActor) created and deployed on actor system $parameterServerAddress")

    //Create work actors (for calculating cost and grads)
    val workersAddress = config.getStringList("estuary.workers")
    val nWorkers = workersAddress.size()
    val nTasksPerWorker = math.ceil(nTasks / nWorkers.toDouble).toInt
    var workerCount = 0
    val workActors = batches.zip(models).zipWithIndex.map { case ((batch, eModel), taskIndex) =>
      val workerIndex = taskIndex / nTasksPerWorker
      workerCount += 1
      system.actorOf(Props(new BatchGradCalculatorActor[M, O](
        batch._1, batch._2, eModel, iteration, getMiniBatches, updateFunc, paramServerActor, opParamsToModelParams
      )).withDeploy(Deploy(scope = RemoteScope(AddressFromURIString(workersAddress.get(workerIndex))))), name=s"worker$workerCount")
    }

    workActors.zipWithIndex.foreach{ case (actor, index) =>
      logger.info(s"${index+1}th Working Actor ($actor) created and deployed on actor system ${workersAddress.get(index / nTasksPerWorker)}")
    }

    //create manager actor (tell workers start to train, handle situations where workers or parameterServer die, storing costHistory)
    val managerSystemAdd = AddressFromURIString(config.getString("estuary.manager"))
    val managerActor = system.actorOf(Props(
      new Manager(paramServerActor, workActors)
    ).withDeploy(Deploy(scope = RemoteScope(managerSystemAdd))), name = "manager")

    logger.info(s"Manager actor ($managerActor) created and deployed on actor system $managerSystemAdd")

    import akka.pattern.ask
    import scala.concurrent.duration._
    implicit val timeout: Timeout = Timeout(100 days) //waiting 100 days in maximum for training

    val trainedParams: Future[Any] = managerActor ? StartTrain //expect to receive trained parameters after sending StartTrain to manager

    logger.info("Waiting for training to be done, please see working actors for training processes...")

    val nowParams = Await.result(trainedParams, 100 days).asInstanceOf[CurrentParams[O]].params

    logger.info("Training complete")

    val costHistoryFuture: Future[List[Double]] = (managerActor ? GetCostHistory).mapTo[List[Double]]
    for (cost <- Await.result(costHistoryFuture, 1 hour)) costHistory += cost

    //Shutdown main system, however, parameterServer actor and all worker actors are not under control of this actor system,
    //hence they will not be terminated.
    system.terminate()

    opParamsToModelParams(nowParams)
  }
}
