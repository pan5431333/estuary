package estuary.demo

import estuary.concurrency.createActorSystem

object CreateActorSystem {
  def main(args: Array[String]): Unit = {
    val name = args(0)
    val ip = args(1)
    val port = args(2).toInt
    createActorSystem(name, ip, port)
  }
}
