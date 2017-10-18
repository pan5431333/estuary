package estuary.components.optimizer

import akka.actor.{Actor, ActorSystem, Props, Terminated}
import breeze.linalg.DenseMatrix
import estuary.components.optimizer.AdamOptimizer.AdamParam
import estuary.components.optimizer.AkkaAdamOptimizer.CostHistory
import estuary.concurrency.AdamBatchGradCalculator.Start
import estuary.concurrency.ParameterServer.{CurrentParams, GetCurrentParams, UpdateParams}
import estuary.concurrency.{AdamBatchGradCalculator, ParameterServer}
import estuary.model.Model
import org.apache.log4j.Logger

class AkkaAdamOptimizer(override val iteration: Int,
                        override val learningRate: Double,
                        override val paramSavePath: String,
                        override val miniBatchSize: Int,
                        override val momentumRate: Double,
                        override val adamRate: Double,
                        val nTasks: Int)
  extends AdamOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate)
    with AbstractDistributed[AdamParam, Seq[DenseMatrix[Double]]] {
  override val logger: Logger = Logger.getLogger(this.getClass)

  private var nowParams: AdamParam = _

  override protected def updateParameterServer(grads: AdamParam, miniBatchTime: Int): Unit = {}

  def parOptimize(feature: DenseMatrix[Double], label: DenseMatrix[Double], model: Model[Seq[DenseMatrix[Double]]], initParams: Seq[DenseMatrix[Double]]): Seq[DenseMatrix[Double]] = {
    val batches = genParBatches(feature, label).seq
    val models = batches.indices.map(_ => model.copyStructure)
    @volatile
    var isDone: Boolean = false
    @volatile
    var isDone2: Boolean = false

    val mainActor = AkkaAdamOptimizer.system.actorOf(Props(new Actor{
      //Create parameter server actor
      private val init = AdamParam(initParams, getInitAdam(initParams), getInitAdam(initParams))
      private val paramServerActor = context.actorOf(Props(new ParameterServer[AdamParam](init)), name="ParameterServer")

      //Create work actors (for calculating cost and grads)
      private val workActors = batches.zip(models).map { case (batch, eModel) =>
        context.actorOf(Props(
          new AdamBatchGradCalculator[Seq[DenseMatrix[Double]], AdamParam](
            batch._1, batch._2, eModel, iteration, getMiniBatches, updateFunc, paramServerActor, _.modelParam)
        ))}

      private var nWorksDead: Int = 0

      override def receive: Actor.Receive = {
        case Start() =>
          //Start every work actors
          workActors foreach {_ ! Start()}
          workActors foreach {a => context.watch(a)}

        case Terminated(_) =>
          nWorksDead += 1
          if (nWorksDead == nTasks) {isDone = true; paramServerActor ! GetCurrentParams}

        case CurrentParams(params) =>
          nowParams = params.asInstanceOf[AdamParam]
          isDone2 = true
          context.stop(self)

        case CostHistory(cost) => addCostHistory(cost)
      }
    }))

    //Start training
    mainActor ! Start()

    //Waiting for training to be done.
    while (!isDone || !isDone2) {}

    //Shutdown main actor, parameterServer actor and all worker actors
    AkkaAdamOptimizer.system.terminate()

    nowParams.modelParam
  }
}

object AkkaAdamOptimizer {
  val system = ActorSystem("AkkaAdamOptimizer")

  def apply(iteration: Int = 100, learningRate: Double = 0.001, paramSavePath: String = System.getProperty("user.dir"), miniBatchSize: Int = 64, momentumRate: Double = 0.9, adamRate: Double = 0.999, nTasks: Int = 4): AkkaAdamOptimizer = {
    new AkkaAdamOptimizer(iteration, learningRate, paramSavePath, miniBatchSize, momentumRate, adamRate, nTasks)
  }

  sealed trait AkkaAdamOptimizerMsg
  case class CostHistory(cost: Double) extends AkkaAdamOptimizerMsg
}


