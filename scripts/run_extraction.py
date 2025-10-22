# scripts/run_extraction.py
import sys
import os
import time

# Ajouter le dossier src au chemin Python
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
src_path = os.path.join(project_root, 'src')

sys.path.insert(0, src_path)

def main():
    print("🚀 LANCEMENT DE L'EXTRACTION DES FEATURES")
    print("=" * 50)
    
    try:
        # Importer et exécuter l'extraction
        from feature_extraction import main as extract_features
        
        start_time = time.time()
        extract_features()
        end_time = time.time()
        
        print(f"\n⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")
        print("✅ Processus terminé avec succès!")
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("💡 Vérifiez que le fichier feature_extraction.py est dans le dossier src/")
        
    except Exception as e:
        print(f"❌ Erreur during l'exécution: {e}")

if __name__ == "__main__":
    main()