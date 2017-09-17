package org.mengpan.deeplearning.components.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}
import org.mengpan.deeplearning.components.layers.Layer

/**Marker interface indicating Heuristic optimizer*/
trait Heuristic extends Optimizer
