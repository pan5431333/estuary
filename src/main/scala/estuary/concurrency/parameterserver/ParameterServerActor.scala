package estuary.concurrency.parameterserver

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import estuary.concurrency.MyMessage
import estuary.concurrency.parameterserver.ParameterServerActor._

class ParameterServerActor extends Actor with ActorLogging{
  private[this] var parameters: AnyRef = _
  private[this] var miniBatchTime: Int = 0

  override def receive: Actor.Receive = {
    case GetCurrentParams => sender ! CurrentParams(parameters)

    case GetCurrentParamsForUpdate => sender ! CurrentParamsForUpdate(parameters, miniBatchTime)

    case UpdateParams(newParams: AnyRef) =>
      parameters = newParams
      miniBatchTime += 1
  }
}

object ParameterServerActor {
  def props: Props = {
    Props(classOf[ParameterServerActor])
  }


  sealed trait ParameterServerActorMsg extends Serializable with MyMessage

  final case object GetCurrentParams extends ParameterServerActorMsg
  final case object GetCurrentParamsForUpdate extends ParameterServerActorMsg
  final case object GetTrainedParams extends ParameterServerActorMsg
  final case object GetCostHistory extends ParameterServerActorMsg

  final case class UpdateParams(newParams: Any) extends ParameterServerActorMsg

  final case class SetWorkActorsRef(workActors: Seq[ActorRef]) extends ParameterServerActorMsg

  final case class CurrentParams(params: Any) extends ParameterServerActorMsg
  final case class CurrentParamsForUpdate(params: Any, miniBatchTime: Int) extends ParameterServerActorMsg

  final case class CostHistory(cost: Double) extends ParameterServerActorMsg
}
