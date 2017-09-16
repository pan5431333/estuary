package org.mengpan.deeplearning.components.optimizer

import org.apache.log4j.Logger

/**
  * Created by mengpan on 2017/9/10.
  */
trait NonHeuristic extends Optimizer{
  override val logger = Logger.getLogger(this.getClass)
}
