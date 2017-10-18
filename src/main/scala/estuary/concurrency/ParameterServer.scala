package estuary.concurrency

import akka.actor.{Actor, ActorLogging}
import estuary.concurrency.ParameterServer.{CurrentParams, GetCurrentParams, UpdateParams}

class ParameterServer[B](initParams: B) extends Actor with ActorLogging{
  private[this] var parameters: B = initParams

  override def receive: Actor.Receive = {
    case GetCurrentParams => sender ! CurrentParams(parameters)

    case UpdateParams(newParams: B) => parameters = newParams
  }
}

object ParameterServer {
  sealed trait ParameterServerMsg
  case object GetCurrentParams extends ParameterServerMsg
  case class UpdateParams[B](newParams: B) extends ParameterServerMsg
  case class CurrentParams[B](params: B) extends ParameterServerMsg

}
