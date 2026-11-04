package com.example.myapplication

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

import android.util.Size
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy

import android.widget.TextView
import android.os.SystemClock

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var detector: YoloDetector

    @Volatile
    private var lastInferenceMs = 0L

    private lateinit var fpsText: TextView
    private var frameCount = 0
    private var lastFpsTime = SystemClock.elapsedRealtime()

    private lateinit var imageView: ImageView

    // Поток, в котором камера вызывает наш analyzer
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    // Область корутин для тяжёлой обработки (детектор)
    private val processingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // true = предыдущий кадр ещё обрабатывается
    private val busy = AtomicBoolean(false)

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Нужно разрешение на камеру", Toast.LENGTH_SHORT).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        fpsText = findViewById(R.id.fpsText)
        imageView = findViewById(R.id.imageView)
        detector = YoloDetector(this)
        permissionLauncher.launch(android.Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(1280, 720),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector((resolutionSelector))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                // Детектор ещё занят предыдущим кадром, этот выбрасываем
                if (!busy.compareAndSet(false, true)) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                // Копируем кадр в Bitmap и сразу освобождаем буфер камеры
                val bitmap = imageProxy.toBitmap()
                val rotation = imageProxy.imageInfo.rotationDegrees
                imageProxy.close()

                processingScope.launch {
                    try {
                        val result = detect(bitmap, rotation)
                        withContext(Dispatchers.Main) {
                            imageView.setImageBitmap(result)
                            frameCount++
                            val now = SystemClock.elapsedRealtime()
                            if (now - lastFpsTime >= 1000){
                                fpsText.text = "FPS: $frameCount | detect: ${lastInferenceMs} ms"
                                frameCount = 0
                                lastFpsTime = now
                            }
                        }
                    } finally {
                        busy.set(false) // можно брать следующий кадр
                    }
                }
            }

            cameraProvider.unbindAll()
            // Preview не подключаем, поэтому камера на экране не видна
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private val displayBitmaps = arrayOfNulls<Bitmap>(2)
    private var displayIndex = 0
    private fun detect(src: Bitmap, rotation: Int): Bitmap {
        val swap = rotation % 180 != 0
        val uw = if (swap) src.height else src.width
        val uh = if (swap) src.width else src.height

        var out = displayBitmaps[displayIndex]
        if (out == null || out.width != uw || out.height != uh){
            out = Bitmap.createBitmap(uw, uh, Bitmap.Config.ARGB_8888)
            displayBitmaps[displayIndex] = out
        }

        displayIndex = displayIndex xor 1

        val t0 = SystemClock.elapsedRealtime()
        val detections = detector.detect(src, rotation)
        lastInferenceMs = SystemClock.elapsedRealtime() - t0

        val matrix = Matrix().apply {
            postTranslate(-src.width / 2f, -src.height / 2f)
            postRotate(rotation.toFloat())
            postTranslate(uw / 2f, uh / 2f)
        }
        Canvas(out).drawBitmap(src, matrix, null)
        drawDetection(out, detections)
        return out
    }

    private fun drawDetection(bitmap: Bitmap, detections: List<Detection>){
        val canvas = Canvas(bitmap)
        val stroke = bitmap.width / 200f
        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = stroke
        }

        val textPaint = Paint().apply {
            color = Color.BLUE
            textSize = bitmap.width / 28f
            isAntiAlias = true
        }

        for (d in detections){
            canvas.drawRect(d.box, boxPaint)
            val label = COCO_LABELS.getOrElse(d.classId) {"id${d.classId}"}
            canvas.drawText(
                    "$label ${(d.score* 100).toInt()}%",
                    d.box.left,
                    (d.box.top - stroke).coerceAtLeast(textPaint.textSize),
                    textPaint
            )

        }
    }

    override fun onDestroy() {
        super.onDestroy()
        processingScope.cancel()
        analysisExecutor.shutdown()
        detector.close()
    }
}