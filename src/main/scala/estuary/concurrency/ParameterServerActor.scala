package estuary.concurrency

import akka.actor.{Actor, ActorLogging, ActorRef, Terminated}
import estuary.concurrency.BatchGradCalculatorActor.StartTrain
import estuary.concurrency.ParameterServerActor._

class ParameterServerActor[B](initParams: B) extends Actor with ActorLogging{
  private[this] var parameters: B = initParams
  private[this] var miniBatchTime: Int = 0
  private[this] var appMainActor: ActorRef = _
  private[this] var workActors: Seq[ActorRef] = _
  private[this] var nWorkActors: Int = _
  private[this] var nWorkActorsDead: Int = _
  private[this] var costHistory: List[Double] = List()

  override def receive: Actor.Receive = {
    case SetWorkActorsRef(workActors) =>
      this.workActors = workActors
      workActors.foreach(context.watch)
      nWorkActors = workActors.length

    case Terminated(_) =>
      nWorkActorsDead += 1
      if (nWorkActorsDead >= nWorkActors) appMainActor ! CurrentParams(parameters)

    case StartTrain => workActors.foreach(_.forward(StartTrain))

    case GetCurrentParams => sender ! CurrentParams(parameters)

    case GetCurrentParamsForUpdate => sender ! CurrentParamsForUpdate(parameters, miniBatchTime)

    case GetTrainedParams => appMainActor = sender()

    case UpdateParams(newParams: B) =>
      parameters = newParams
      miniBatchTime += 1

    case CostHistory(cost) => costHistory = cost :: costHistory

    case GetCostHistory => sender ! costHistory.reverse
  }
}

object ParameterServerActor {
  sealed trait ParameterServerActorMsg extends Serializable

  final case object GetCurrentParams extends ParameterServerActorMsg
  final case object GetCurrentParamsForUpdate extends ParameterServerActorMsg
  final case object GetTrainedParams extends ParameterServerActorMsg
  final case object GetCostHistory extends ParameterServerActorMsg

  final case class UpdateParams[B](newParams: B) extends ParameterServerActorMsg

  final case class SetWorkActorsRef(workActors: Seq[ActorRef]) extends ParameterServerActorMsg

  final case class CurrentParams[B](params: B) extends ParameterServerActorMsg
  final case class CurrentParamsForUpdate[B](params: B, miniBatchTime: Int) extends ParameterServerActorMsg

  final case class CostHistory(cost: Double) extends ParameterServerActorMsg
}
