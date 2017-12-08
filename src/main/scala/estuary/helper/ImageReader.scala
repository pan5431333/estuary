package estuary.helper

import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.DenseVector
import org.apache.log4j.Logger

/**
  * Created by mengpan on 2017/9/10.
  */
object ImageReader {
  val logger: Logger = Logger.getLogger(this.getClass)

  def readImageToRGBVector(fileName: String): Option[DenseVector[Double]] = {
    try {
      val image = ImageIO.read(new File(fileName))

      val (red, greenAndBlue) = (image.getMinX until image.getWidth).par.flatMap { width =>
        (image.getMinY until image.getHeight).par.map { height =>
          val pixel = image.getRGB(width, height)
          (((pixel & 0xff0000) >> 16).toDouble, ((pixel & 0xff00) >> 8).toDouble, (pixel & 0xff).toDouble)
        }
      }
        .toList
        .unzip[Double, (Double, Double)](f => (f._1, (f._2, f._3)))

      val (green, blue) = greenAndBlue
        .unzip[Double, Double](asPair => (asPair._1, asPair._2))

      val flattenFeature = red ::: green ::: blue ::: Nil

      val resVector = DenseVector(flattenFeature.toArray)
      Some(resVector)
    } catch {
      case _: Exception => None
    }
  }

}
