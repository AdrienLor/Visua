fn main() {
 
     #[cfg(windows)]
    {
        use std::path::Path;
        let ico = "assets/app.ico";
        let rc  = "assets/windows/app.rc";

        assert!(Path::new(ico).exists(), "Fichier ICO introuvable: {ico}");
        assert!(Path::new(rc).exists(),  "Fichier RC introuvable:  {rc}");

        embed_resource::compile(rc, embed_resource::NONE);
     
    }
}