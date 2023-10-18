import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_text_extraction/camera.dart';
import 'package:tflite_text_extraction/helper.dart';

void main() async {
  List<CameraDescription> cameras = <CameraDescription>[];

  try {
    WidgetsFlutterBinding.ensureInitialized();
    cameras = await availableCameras();
  } on CameraException catch (e) {
    _logError(e.code, e.description);
  }
  runApp(MyApp(
    cameras: cameras,
  ));
}

void _logError(String code, String? message) {
  // ignore: avoid_print
  print('Error: $code${message == null ? '' : '\nError Message: $message'}');
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const MyApp({super.key, required this.cameras});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        theme: ThemeData(
          useMaterial3: true,
        ),
        home: HomeScreen(
          cameras: cameras,
        ));
  }
}

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const HomeScreen({super.key, required this.cameras});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late Image _imageFile;
  late ImageHelper _imageHelper;

  @override
  void initState() {
    super.initState();

    _imageHelper = ImageHelper();
    _imageHelper.init();
    _imageFile = Image.asset('assets/wizardiusbewebicon.png');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor:
            ColorScheme.fromSeed(seedColor: Colors.deepPurple).inversePrimary,
        title: const Text('Flutter Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _imageFile,
            ElevatedButton(
                onPressed: () {
                  Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (context) => CameraScreen(widget.cameras,
                                  imageFile: (imageFile) {
                                _imageProcess((imageFile));
                              })));
                },
                child: const Text('Open Camera')),
          ],
        ),
      ),
    );
  }

  Future<void> _imageProcess(XFile imageFile) async {
    var outputFile = await _imageHelper.analyzeImage(imageFile);

    setState(() {
      _imageFile = Image.memory(outputFile);
    });
  }
}
