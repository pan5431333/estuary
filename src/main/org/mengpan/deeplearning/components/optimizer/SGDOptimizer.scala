package org.mengpan.deeplearning.components.optimizer

/**
  * Created by mengpan on 2017/9/9.
  */
class SGDOptimizer extends Optimizer with MiniBatchable with NonHeuristic

object SGDOptimizer {
  def apply(miniBatchSize: Int): SGDOptimizer = {
    new SGDOptimizer()
      .setMiniBatchSize(miniBatchSize)
  }
}
