public class BioPhoto {

    private String fileName;
    private float[] feature;

    public BioPhoto() {
    }

    public BioPhoto(String fileName, float[] feature) {
        this.fileName = fileName;
        this.feature = feature;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public float[] getFeature() {
        return feature;
    }

    public void setFeature(float[] feature) {
        this.feature = feature;
    }
}
