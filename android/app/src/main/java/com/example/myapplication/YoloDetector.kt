package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.min

import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

data class Detection(val box: RectF, val score: Float, val classId: Int)

class YoloDetector(
    context: Context,
    modelName: String = "yolo26n.tflite",
    private val inputSize: Int = 416, // ----------
    private val threshold: Float = 0.35f
) {
    private val interpreter: Interpreter

    private var gpuDelegate: GpuDelegate? = null

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())

    private val floatBuf = inputBuffer.asFloatBuffer()
    private val pixels = IntArray(inputSize * inputSize)
    private val letterBoxBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val letterBoxCanvas = Canvas(letterBoxBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    private val output = Array(1){Array(300){ FloatArray(6) } }

    init {
        val options = Interpreter.Options()
        var usingGpu = false

        CompatibilityList().use {compat ->
            if (compat.isDelegateSupportedOnThisDevice){
                try {
                    val gpuOption = compat.bestOptionsForThisDevice.apply {
                        setSerializationParams(
                            context.cacheDir.absolutePath,
                            "yolo26n_416_v1"
                        )
                    }
                    gpuDelegate = GpuDelegate(gpuOption)
                    options.addDelegate(gpuDelegate)
                    usingGpu = true
                } catch (e: Exception){
                    android.util.Log.e("YOLO", "GPU не запустился", e)
                    gpuDelegate?.close()
                    gpuDelegate = null
                }
            }

        }

        if (!usingGpu) options.setNumThreads(4)
        android.util.Log.d("YOLO", "GPU delegate: $usingGpu")

        interpreter = Interpreter(loadModel(context, modelName), options)

        val shape = interpreter.getOutputTensor(0).shape()
        val inshape = interpreter.getInputTensor(0).shape()

        require(shape.size == 3 && shape[1] == 300 && shape[2] == 6){
            "Неожиданная форма выхода"
        }
        require(inshape.contentEquals(intArrayOf(1, 3, inputSize, inputSize))){
            "Неожиданный размер изображения"
        }
    }
    private fun loadModel(context: Context, name: String): ByteBuffer {
        val fd = context.assets.openFd(name)
        FileInputStream(fd.fileDescriptor).use {stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    }

    fun detect(src: Bitmap, rotation: Int): List<Detection>{
        val swap = rotation % 180 != 0
        val uw = if (swap) src.height else src.width
        val uh = if (swap) src.width else src.height

        val scale = min(inputSize/uw.toFloat(), inputSize/uh.toFloat())
        val padX = (inputSize - uw*scale) / 2f
        val padY = (inputSize - uh*scale) / 2f // -----------

        letterBoxCanvas.drawColor(Color.rgb(114, 114, 114))
        val matrix = Matrix().apply {
            postTranslate(-src.width / 2f, -src.height / 2f)
            postRotate(rotation.toFloat())
            postScale(scale, scale)
            postTranslate(inputSize / 2f, inputSize / 2f)
        }
        letterBoxCanvas.drawBitmap(src, matrix, paint)

        letterBoxBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val n = inputSize * inputSize
        for (i in 0 until n) {
            val p = pixels[i]
            floatBuf.put(i, ((p shr 16) and 0xFF) / 255f)
            floatBuf.put(n+i, ((p shr 8) and 0xFF) / 255f)
            floatBuf.put(2*n+i, (p and 0xFF) / 255f)
        }
        inputBuffer.rewind()

        interpreter.run(inputBuffer, output)

        val result = ArrayList<Detection>()
        for (row in output[0]){
            if (row[5].toInt() != 0) continue

            val score = row[4]
            if (score < threshold) continue

            var x1 = row[0]; var y1 = row[1]; var x2 = row[2]; var y2 = row[3]

            val box = RectF(
                ((x1 - padX) / scale).coerceIn(0f, src.width.toFloat()),
                ((y1 - padY) / scale).coerceIn(0f, src.height.toFloat()),
                ((x2 - padX) / scale).coerceIn(0f, src.width.toFloat()),
                ((y2 - padY) / scale).coerceIn(0f, src.height.toFloat())
            )

            result.add(Detection(box, score, row[5].toInt()))
        }
        return result
    }
    fun close(){
        interpreter.close()
        gpuDelegate?.close()
    }
}

val COCO_LABELS = listOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
)