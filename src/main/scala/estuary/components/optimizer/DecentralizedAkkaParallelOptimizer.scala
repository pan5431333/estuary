package estuary.components.optimizer

import java.io.FileNotFoundException

import akka.actor.{ActorRef, AddressFromURIString, Deploy, PoisonPill, Props}
import akka.remote.RemoteScope
import akka.util.Timeout
import com.typesafe.config.{Config, ConfigFactory}
import estuary.concurrency.createActorSystem
import estuary.concurrency.decentralized.DecentralizedBatchCalculator.{CurrentParam, Neibours}
import estuary.concurrency.decentralized.Manager.{GetCostHistory, StartTrain}
import estuary.concurrency.decentralized.{DecentralizedBatchCalculator, Manager}
import estuary.data.Reader
import estuary.model.Model

import scala.collection.mutable
import scala.concurrent.Await

/**
  * Created by mengpan on 2017/10/28.
  */
trait DecentralizedAkkaParallelOptimizer[O <: AnyRef, M <: AnyRef] extends AbstractAkkaParallelOptimizer[O, M] {

  protected def avgOpFunc(params: Seq[O]): O

  /**
    * Optimize the model in parallel, and returning the trained parameters with the same dimensions of initParams.
    * The method parameter 'model' is used here to create several model instances (with copyStructure() method), and
    * then they are distributed to different threads or machines.
    *
    * @param model      an instance of trait Model, used to create many copies and then distribute them to different threads
    *                   or machines.
    * @return trained parameters, with same dimension with the given initial parameters.
    */
  override def parOptimize(model: Model[M]): M = {

    //Read configuration file "estuary.conf"
    val config = ConfigFactory.load("estuary")

    //get Workers config
    val workers = config.getConfigList("estuary.workers").toArray

    //get manager config
    val managerAddr = AddressFromURIString(config.getString("estuary.manager"))

    //create application's MainSystem
    val system = try {
      createActorSystem("MainSystem", "MainSystem")
    } catch {
      case _: FileNotFoundException =>
        logger.warn("No configuration file MainSystem found, use default akka system: akka.tcp://MainSystem@127.0.0.1:2552")
        createActorSystem("MainSystem")
    }

    logger.info("MainSystem {} created", system)

    //create model instances
    val nWorkers = workers.length
    val models = (0 to nWorkers).par.map(_ => model.copyStructure)

    //mapping id to workers' actor reference
    val workersId = new scala.collection.mutable.HashMap[Long, ActorRef]()

    //mapping neibours' id to workers' actor reference
    val workersNeibour = new mutable.HashMap[ActorRef, Seq[Long]]()

    //create all workers
    var workerCnt = 0
    val allWorkers = for ((worker, model) <- workers.zip(models)) yield {
      workerCnt += 1
      val config_ = worker.asInstanceOf[Config]
      val id = config_.getString("id").toLong
      val addr = AddressFromURIString(config_.getString("address"))
      val filePath = config_.getString("file-path")
      val neibour = config_.getLongList("neibours").toArray.map(_.asInstanceOf[Long])
      val dataReader = Class.forName(config_.getString("data-reader")).getConstructor().newInstance().asInstanceOf[Reader]
      val actor = system.actorOf(DecentralizedBatchCalculator.props(id, filePath, dataReader, model, iteration, getMiniBatches, updateFunc, modelParamsToOpParams, opParamsToModelParams, avgOpFunc)
        .withDeploy(Deploy(scope = RemoteScope(addr))), name = s"worker$workerCnt")
      workersId.+=(id -> actor)
      workersNeibour.+=(actor -> neibour)
      actor
    }

    //send neibours' actor references to every worker
    for (worker <- allWorkers) {
      val neibourMsg = Neibours(workersNeibour(worker).map(workersId(_)))
      worker ! neibourMsg
    }

    allWorkers.zipWithIndex.foreach { case (actor, index) =>
      logger.info(s"${index + 1}th Working Actor ($actor) created")
    }

    //create manager
    val manager = system.actorOf(Manager.props(allWorkers).withDeploy(Deploy(scope = RemoteScope(managerAddr))))

    logger.info(s"Manager actor ($manager) created and deployed on actor system $managerAddr")

    import akka.pattern._
    import scala.concurrent.duration._
    implicit val timeout = Timeout(100 days)
    val futureParams = manager ? StartTrain
    logger.info("Waiting for training to be done, please see working actors for training progress...")
    val trainedParams = Await.result(futureParams, 100 days).asInstanceOf[CurrentParam].param
    logger.info("Training complete")

    val futureCostHistory = (manager ? GetCostHistory).mapTo[List[Double]]
    for (cost <- Await.result(futureCostHistory, 1 hour)) costHistory += cost

    //shutdown workers and manager
    manager ! PoisonPill
    allWorkers.foreach(_ ! PoisonPill)

    system.terminate()
    opParamsToModelParams(trainedParams.asInstanceOf[O])
  }

}
