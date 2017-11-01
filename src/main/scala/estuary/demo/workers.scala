package estuary.demo

import estuary.concurrency.createActorSystem

object worker1 extends App{
  createActorSystem("Worker1", "192.168.131.1", 2553)
}

object worker2 extends App{
  createActorSystem("Worker2", "192.168.131.1", 2554)
}

object worker3 extends App{
  createActorSystem("Worker2", "192.168.131.1", 2555)
}

object worker4 extends App{
  createActorSystem("Worker2", "192.168.131.1", 2556)
}
