import jcuda.Pointer
import jcuda.runtime._
import jcuda.Sizeof

object JCUDAExample {

  def main(args: Array[String]): Unit = {
    // Инициализация CUDA (если есть несколько устройств, нужно выбрать нужное)
    JCuda.cudaSetDevice(0)

    // Данные на хосте (CPU)
    val hostData = Array(1.0f, 2.0f, 3.0f, 4.0f)

    // Выделение памяти на GPU
    val deviceData = new Pointer()
    JCuda.cudaMalloc(deviceData, hostData.length * Sizeof.FLOAT)

    // Копирование данных с CPU на GPU
    JCuda.cudaMemcpy(deviceData, Pointer.to(hostData), hostData.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice)

    // Копирование данных с GPU обратно на CPU
    val result = new Array[Float](hostData.length)
    JCuda.cudaMemcpy(Pointer.to(result), deviceData, hostData.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)

    // Выводим результат
    println("Data copied from GPU to CPU:")
    result.foreach(println)

    // Освобождение памяти
    JCuda.cudaFree(deviceData)
  }
}
