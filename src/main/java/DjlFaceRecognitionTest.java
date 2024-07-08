import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.collections4.ListUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class DjlFaceRecognitionTest implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(DjlFaceRecognitionTest.class);

    private static final float featureThresholdForMatch = 0.90f;
    private static final int numberOfImagesPerThread = 100;
    private Criteria<Image, float[]> criteria;
    private ZooModel<Image, float[]> model;
    private Predictor<Image, float[]> predictor;


    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DjlFaceRecognitionTest djlFaceRecognitionTest = new DjlFaceRecognitionTest();
        djlFaceRecognitionTest.doTest();
    }

    public DjlFaceRecognitionTest() throws IOException, ModelException {
        criteria = buildCriteria();
        model = loadZooModel(criteria);
        predictor = createPredictor(model);
    }

    public void doTest() {
        try {
            // Build set of photos
            List<BioPhoto> bioPhotoDataSet = getBioPhotoDataSet();

            // Load sample Photo
            URL sampleBioPhotoUrl = DjlFaceRecognitionTest.class.getClassLoader().getResource("Sample_Haza7_672x672.jpg");
            File sampleBioPhotoFile = new File(sampleBioPhotoUrl.getPath());
            BioPhoto sampleBioPhoto = loadBioPhoto(sampleBioPhotoFile, predictor);
            BioPhoto matchedBioPhoto = getBioPhotoMatch(sampleBioPhoto, bioPhotoDataSet);
            if (matchedBioPhoto != null) {
                logger.info("Sample biophoto " + sampleBioPhoto.getFileName() + " matches " + matchedBioPhoto.getFileName());
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Returns the biophoto that makes match with the sampleBioPhoto passed as parameter.
     * Returns null if sampleBioPhoto is not found in the dataset
     * @param sampleBioPhoto The sample photo to compare against the data set
     * @param bioPhotos The dataset
     * @return
     */
    private BioPhoto getBioPhotoMatch(BioPhoto sampleBioPhoto, List<BioPhoto> bioPhotos) {
        List<BioPhotoMatch> matchedPhotos = new ArrayList<>();
        for(BioPhoto bioPhoto: bioPhotos) {
            float result = calculateSimilitude(sampleBioPhoto.getFeature(), bioPhoto.getFeature());
            logger.info("Comparing against " + bioPhoto.getFileName() + " similitude " + result);
            if (result >= featureThresholdForMatch) {
                logger.info("Match with " + bioPhoto.getFileName() + " at similitude " + result);
                matchedPhotos.add(new BioPhotoMatch(result, bioPhoto));
            }
        }

        if (matchedPhotos.isEmpty()) {
            logger.info("Not match");
            return null;
        }

        if (matchedPhotos.size() == 1) {
            logger.info("Single match");
            return matchedPhotos.get(0).getBioPhoto();
        }

        // Resolve collision
        logger.info("Multiple match, resolving collision");
        return matchedPhotos
                .stream()
                .sorted()
                .findFirst()
                .map(matchedPhoto -> matchedPhoto.getBioPhoto())
                .orElse(null);
    }

    // Load All BioPhoto pictures
    private List<BioPhoto> getBioPhotoDataSet() throws IOException, InterruptedException, ExecutionException {
        ObjectMapper objectMapper = new ObjectMapper();
        List<BioPhoto> bioPhotos;
        URL featuresFileUrl = DjlFaceRecognitionTest.class.getClassLoader().getResource("features_subset2.json");
        if (featuresFileUrl != null) {
            logger.info("Using features from features file");
            bioPhotos = objectMapper.readValue(new File(featuresFileUrl.getPath()), new TypeReference<List<BioPhoto>>(){});
        } else {// Build digest from source files
            logger.info("Building features file");
            long time = System.currentTimeMillis();
            bioPhotos = getBioPhotoDataSetFromSource();
            objectMapper.writeValue(new File("src/main/resources/features_subset2.json"), bioPhotos);
            logger.info("Loading BioPhotos dataset from source took " + (System.currentTimeMillis()-time) + " ms");
        }

        return bioPhotos;
    }

    private List<BioPhoto> getBioPhotoDataSetFromSource() throws InterruptedException, ExecutionException {
        URL bioPhotosDirPath = DjlFaceRecognitionTest.class.getClassLoader().getResource("surfingtimebiophotos_subset2");
        File folder = new File(bioPhotosDirPath.getPath());
        File[] listOfFiles = folder.listFiles();
        List<BioPhoto> bioPhotos = new ArrayList<>();

        List<CompletableFuture<List<BioPhoto>>> futures = new ArrayList<>();
        List<List<File>> bioPhotoFiles = ListUtils.partition(Arrays.asList(listOfFiles), numberOfImagesPerThread);
        for (List<File> bioPhotoFilesChunk: bioPhotoFiles) {
            // Split into multiple threads
            CompletableFuture<List<BioPhoto>> future = CompletableFuture.supplyAsync(() -> loadBioPhotos(bioPhotoFilesChunk, predictor));
            futures.add(future);
        }

        // Wait for all the processing to complete async
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[futures.size()]));

        // Recover the BioPhotos
        for (CompletableFuture<List<BioPhoto>> completedFuture: futures) {
            List<BioPhoto> completedBioPhotos = completedFuture.get();
            bioPhotos.addAll(completedBioPhotos);
        }

        return bioPhotos;
    }

    private List<BioPhoto> loadBioPhotos(List<File> bioPhotoFiles, Predictor<Image, float[]> predictor) {
        logger.info("Loading image batch of " + bioPhotoFiles.size() + " from source");
        List<BioPhoto> bioPhotos = new ArrayList<>();
        for(File bioPhotoFile: bioPhotoFiles) {
            try {
                BioPhoto bioPhoto = loadBioPhoto(bioPhotoFile, predictor);
                bioPhotos.add(bioPhoto);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }

        logger.info("Batch of " + bioPhotoFiles.size() + " biophotos finished");
        return bioPhotos;
    }

    private BioPhoto loadBioPhoto(File bioPhotoFile, Predictor<Image, float[]> predictor) throws IOException, TranslateException {
        //logger.info("Loading image " + bioPhotoFile.getName());
        //long time = System.currentTimeMillis();
        Image img = ImageFactory.getInstance().fromFile(bioPhotoFile.toPath());
        img.getWrappedImage();
        float[] feature = predictor.predict(img);
        //logger.info("Loading image " + bioPhotoFile.getName() + " took " + (System.currentTimeMillis()-time) + " ms");
        return new BioPhoto(bioPhotoFile.getName(), feature);
    }

    private Criteria<Image, float[]> buildCriteria() {
        logger.info("Building Criteria face_feature with Engine PyTorch");
        long time = System.currentTimeMillis();
        URL model = DjlFaceRecognitionTest.class.getClassLoader().getResource("face_feature.zip");
        Criteria<Image, float[]> criteria =
                Criteria.builder()
                        .setTypes(Image.class, float[].class)
                        .optModelPath(new File(model.getPath()).toPath())
                        .optModelName("face_feature") // specify model file prefix
                        .optTranslator(new FaceFeatureTranslator())
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch") // Use PyTorch engine
                        .build();

        logger.info("Loading criteria took " + (System.currentTimeMillis()-time) + " ms");
        return criteria;
    }

    private ZooModel<Image, float[]> loadZooModel(Criteria<Image, float[]> criteria) throws IOException, ModelException {
        logger.info("Loading ZooModel");
        long time = System.currentTimeMillis();
        ZooModel<Image, float[]> model = criteria.loadModel();
        logger.info("Loading ZooModel took " + (System.currentTimeMillis()-time) + " ms");
        return model;
    }

    private Predictor<Image, float[]> createPredictor(ZooModel<Image, float[]> model) {
        logger.info("Create Predictor");
        long time = System.currentTimeMillis();
        Predictor<Image, float[]> predictor = model.newPredictor();
        logger.info("Create Predicto took " + (System.currentTimeMillis()-time) + " ms");
        return predictor;
    }

    private static float calculateSimilitude(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return (float) ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
    }

    @Override
    public void close() throws Exception {
        predictor.close();
    }

    private static final class FaceFeatureTranslator implements Translator<Image, float[]> {

        FaceFeatureTranslator() {}

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            Pipeline pipeline = new Pipeline();
            pipeline
                    // .add(new Resize(160))
                    .add(new ToTensor())
                    .add(
                            new Normalize(
                                    new float[] {127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f},
                                    new float[] {
                                            128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f
                                    }));

            return pipeline.transform(new NDList(array));
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            float[][] embeddings =
                    result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
            float[] feature = new float[embeddings.length];
            for (int i = 0; i < embeddings.length; i++) {
                feature[i] = embeddings[i][0];
            }
            return feature;
        }
    }
}
