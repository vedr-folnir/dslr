import csv
import numpy as np


def open_dataset(path: str):
    """ 
        ouvre le dataset du path
        recup les infos dans un bloc divise par matiere
        remplace les cases vides par None
        ne prend pas en compte les erreurs
    """
    
    csvfile = open(path, newline='')
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    for i, line in enumerate(file):
        if i == 0:
            dataset = [[] for _ in range(len(line))]
            pass
        for y, elem in enumerate(line):
            if len(elem) == 0:
                dataset[y].append(None)
                continue
            dataset[y].append(elem)
    return dataset


def sort_by_kind(dataset, index):
    """
        prend un dataset et les tri par rapport au donner du champs
        ex: je veux trier par maisons tu donne l'index du champ maison
            le prog cherche le nombre de diff maisons et les trie
            dataset => maison[dataset,Ravenclaw,Slytherin,Gryffindor,Hufflepuff]
        le retour est un tableau de taille variable dans un ordre radom a cause du set()
        avec comme 0 le dataset de base
    """
    names = list(set(dataset[index][1:]))
    sorted = [[] for _ in range(len(names))]
    for stud in range(len(dataset[index])):
        # print(i)
        if dataset[index][stud] in names:
            sorted[names.index(dataset[index][stud])].append(stud)
            # print(stud)
            
        
    return sorted, names

def get_data(dataset, index, to):
    """
        retourne une liste de valeurs du dataset a l'index de chaque membre de to
    """
    values = []
    for elem in to:
        values.append(dataset[index][elem])
    return values


def sigmoid(z):
    """
    Fonction sigmoïde pour la régression logistique
    """
    # Clip pour éviter les overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def prepare_data(dataset, feature_indices, target_index):
    """
    Prépare les données pour l'entraînement
    - dataset: le dataset complet
    - feature_indices: indices des colonnes à utiliser comme features
    - target_index: index de la colonne target (maisons)
    """
    X = []
    y = []
    
    # Récupérer les noms des maisons uniques
    houses = list(set(dataset[target_index][1:]))  # Skip header
    house_to_int = {house: i for i, house in enumerate(houses)}
    
    for i in range(1, len(dataset[0])):  # Skip header row
        # Vérifier que toutes les features sont disponibles
        row_features = []
        skip_row = False
        
        for feat_idx in feature_indices:
            if dataset[feat_idx][i] is None or dataset[feat_idx][i] == '':
                skip_row = True
                break
            try:
                row_features.append(float(dataset[feat_idx][i]))
            except ValueError:
                skip_row = True
                break
        
        # Vérifier que le target est disponible
        if dataset[target_index][i] is None or dataset[target_index][i] == '':
            skip_row = True
        
        if not skip_row:
            X.append(row_features)
            y.append(house_to_int[dataset[target_index][i]])
    
    return np.array(X), np.array(y), houses, house_to_int


def normalize_features(X):
    """
    Normalise les features (standardisation)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Éviter la division par zéro
    std = np.where(std == 0, 1, std)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def one_vs_all_logistic_regression(X, y, num_classes, learning_rate=0.01, epochs=1000):
    """
    Régression logistique One-vs-All pour classification multi-classe
    """
    n_samples, n_features = X.shape
    
    # Initialiser les poids pour chaque classe
    weights = np.random.normal(0, 0.01, (num_classes, n_features))
    biases = np.zeros(num_classes)
    
    # Historique des coûts
    cost_history = []
    
    for epoch in range(epochs):
        # Mélanger les données pour SGD
        indices = np.random.permutation(n_samples)
        epoch_cost = 0
        
        for i in indices:
            x_i = X[i]
            y_i = y[i]
            
            # Pour chaque classe (One-vs-All)
            for class_idx in range(num_classes):
                # Target binaire: 1 si c'est la classe courante, 0 sinon
                y_binary = 1 if y_i == class_idx else 0
                
                # Forward pass
                z = np.dot(x_i, weights[class_idx]) + biases[class_idx]
                prediction = sigmoid(z)
                
                # Calculer le coût (log-loss)
                cost = -y_binary * np.log(prediction + 1e-15) - (1 - y_binary) * np.log(1 - prediction + 1e-15)
                epoch_cost += cost
                
                # Backward pass (gradients)
                error = prediction - y_binary
                dw = error * x_i
                db = error
                
                # Mise à jour des poids (SGD)
                weights[class_idx] -= learning_rate * dw
                biases[class_idx] -= learning_rate * db
        
        # Enregistrer le coût moyen de l'époque
        avg_cost = epoch_cost / (n_samples * num_classes)
        cost_history.append(avg_cost)
        
        # Afficher le progrès
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {avg_cost:.4f}")
    
    return weights, biases, cost_history


def predict(X, weights, biases):
    """
    Faire des prédictions avec le modèle entraîné
    """
    n_samples = X.shape[0]
    num_classes = weights.shape[0]
    
    # Calculer les scores pour chaque classe
    scores = np.zeros((n_samples, num_classes))
    for class_idx in range(num_classes):
        z = np.dot(X, weights[class_idx]) + biases[class_idx]
        scores[:, class_idx] = sigmoid(z)
    
    # Prendre la classe avec le score le plus élevé
    predictions = np.argmax(scores, axis=1)
    return predictions, scores


def calculate_accuracy(y_true, y_pred):
    """
    Calculer la précision
    """
    return np.mean(y_true == y_pred)


def train_hogwarts_classifier(dataset):
    """
    Entraîner un classificateur pour les maisons de Poudlard
    """
    print("=== ENTRAÎNEMENT DU CLASSIFICATEUR POUDLARD ===")
    
    # Utiliser plus de features pour améliorer la précision
    # Toutes les matières importantes basées sur l'analyse
    feature_indices = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # Toutes les matières
    feature_names = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", 
                    "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
                    "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
    target_index = 1  # Index de la colonne des maisons
    
    # Préparer les données
    print("Préparation des données...")
    X, y, houses, house_to_int = prepare_data(dataset, feature_indices, target_index)
    print(f"Données préparées: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"Features utilisées: {feature_names}")
    print(f"Maisons: {houses}")
    
    # Normaliser les features
    X_norm, mean, std = normalize_features(X)
    
    # Diviser en train/validation (85/15 pour plus de données d'entraînement)
    split_idx = int(0.85 * len(X))
    indices = np.random.permutation(len(X))
    
    X_train = X_norm[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_val = X_norm[indices[split_idx:]]
    y_val = y[indices[split_idx:]]
    
    print(f"Train set: {len(X_train)} échantillons")
    print(f"Validation set: {len(X_val)} échantillons")
    
    # Entraîner le modèle avec hyperparamètres optimisés
    print("\nEntraînement en cours...")
    weights, biases, cost_history = one_vs_all_logistic_regression(
        X_train, y_train, 
        num_classes=len(houses),
        learning_rate=0.3,  # Learning rate plus élevé
        epochs=2000         # Plus d'époques
    )
    
    # Évaluer sur le set de validation
    print("\nÉvaluation...")
    y_pred_train, _ = predict(X_train, weights, biases)
    y_pred_val, scores_val = predict(X_val, weights, biases)
    
    train_acc = calculate_accuracy(y_train, y_pred_train)
    val_acc = calculate_accuracy(y_val, y_pred_val)
    
    print(f"Précision sur l'entraînement: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Précision sur la validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Vérifier si on atteint l'objectif de 98%
    if val_acc < 0.98:
        print(f"⚠️  Objectif de 98% non atteint (actuel: {val_acc*100:.2f}%)")
        print("Tentative avec des paramètres plus agressifs...")
        
        # Réentraîner avec des paramètres plus agressifs
        weights, biases, cost_history = one_vs_all_logistic_regression(
            X_train, y_train, 
            num_classes=len(houses),
            learning_rate=0.5,
            epochs=3000
        )
        
        y_pred_train, _ = predict(X_train, weights, biases)
        y_pred_val, scores_val = predict(X_val, weights, biases)
        
        train_acc = calculate_accuracy(y_train, y_pred_train)
        val_acc = calculate_accuracy(y_val, y_pred_val)
        
        print(f"Nouvelle précision sur l'entraînement: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Nouvelle précision sur la validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    if val_acc >= 0.98:
        print(f"✅ Objectif de 98% atteint ! ({val_acc*100:.2f}%)")
    
    # Afficher quelques prédictions
    print("\n=== EXEMPLES DE PRÉDICTIONS ===")
    for i in range(min(10, len(X_val))):
        true_house = houses[y_val[i]]
        pred_house = houses[y_pred_val[i]]
        confidence = scores_val[i][y_pred_val[i]]
        status = "✅" if true_house == pred_house else "❌"
        print(f"{status} Vrai: {true_house:12} | Prédit: {pred_house:12} | Confiance: {confidence:.3f}")
    
    return weights, biases, mean, std, houses, house_to_int, feature_names


def save_model(weights, biases, mean, std, houses, house_to_int, feature_names, filename="model.npz"):
    """
    Sauvegarder le modèle entraîné
    """
    np.savez(filename, 
             weights=weights, 
             biases=biases, 
             mean=mean, 
             std=std, 
             houses=houses, 
             house_to_int=list(house_to_int.items()),
             feature_names=feature_names)
    print(f"Modèle sauvegardé dans {filename}")
    
    # Sauvegarder aussi en format lisible
    weights_readable_file = filename.replace('.npz', '_weights.txt')
    save_weights_readable(weights, biases, houses, feature_names, weights_readable_file)


def save_weights_readable(weights, biases, houses, feature_names, filename="weights.txt"):
    """
    Sauvegarder les poids dans un format lisible
    """
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("      POIDS DU MODÈLE RÉGRESSION LOGISTIQUE POUDLARD\n")
        f.write("="*60 + "\n\n")
        
        f.write("Features utilisées:\n")
        for i, feature in enumerate(feature_names):
            f.write(f"  {i:2d}: {feature}\n")
        f.write("\n")
        
        f.write("POIDS PAR MAISON:\n")
        f.write("-" * 60 + "\n")
        
        for house_idx, house in enumerate(houses):
            f.write(f"\n🏠 {house.upper()}:\n")
            f.write(f"   Biais: {biases[house_idx]:8.4f}\n")
            f.write("   Poids des matières:\n")
            
            # Trier les poids par ordre d'importance (valeur absolue)
            weight_importance = [(abs(weights[house_idx][i]), i, weights[house_idx][i]) 
                               for i in range(len(feature_names))]
            weight_importance.sort(reverse=True)
            
            for abs_weight, feat_idx, weight in weight_importance:
                influence = "📈 Très forte" if abs_weight > 2 else "📊 Forte" if abs_weight > 1 else "📉 Moyenne" if abs_weight > 0.5 else "📋 Faible"
                sign = "➕" if weight > 0 else "➖"
                f.write(f"     {sign} {feature_names[feat_idx]:25}: {weight:8.4f} ({influence})\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRÉTATION:\n")
        f.write("➕ Poids positif = Plus cette matière a une note élevée, plus l'étudiant\n")
        f.write("   a de chances d'être dans cette maison\n")
        f.write("➖ Poids négatif = Plus cette matière a une note élevée, moins l'étudiant\n")
        f.write("   a de chances d'être dans cette maison\n")
        f.write("="*60 + "\n")
    
    print(f"Poids sauvegardés en format lisible: {filename}")


def display_weights(weights, biases, houses, feature_names):
    """
    Afficher les poids de manière lisible dans la console
    """
    print("\n" + "="*60)
    print("           ANALYSE DES POIDS DU MODÈLE")
    print("="*60)
    
    for house_idx, house in enumerate(houses):
        print(f"\n🏠 {house.upper()}:")
        print(f"   Biais: {biases[house_idx]:8.4f}")
        print("   Top 5 matières les plus influentes:")
        
        # Trier par importance
        weight_importance = [(abs(weights[house_idx][i]), i, weights[house_idx][i]) 
                           for i in range(len(feature_names))]
        weight_importance.sort(reverse=True)
        
        for i, (abs_weight, feat_idx, weight) in enumerate(weight_importance[:5]):
            sign = "➕" if weight > 0 else "➖"
            print(f"     {i+1}. {sign} {feature_names[feat_idx]:25}: {weight:8.4f}")


def load_model(filename="model.npz"):
    """
    Charger un modèle pré-entraîné
    """
    data = np.load(filename, allow_pickle=True)
    weights = data['weights']
    biases = data['biases']
    mean = data['mean']
    std = data['std']
    houses = data['houses'].tolist()
    house_to_int = dict(data['house_to_int'].tolist())
    feature_names = data['feature_names'].tolist() if 'feature_names' in data else ["Feature_" + str(i) for i in range(weights.shape[1])]
    print(f"Modèle chargé depuis {filename}")
    return weights, biases, mean, std, houses, house_to_int, feature_names


def predict_test_file(test_file_path, model_filename="model.npz", output_file="houses.csv"):
    """
    Faire des prédictions sur un fichier de test et sauvegarder les résultats
    """
    print(f"=== PRÉDICTION SUR {test_file_path} ===")
    
    # Charger le modèle
    weights, biases, mean, std, houses, house_to_int, feature_names = load_model(model_filename)
    
    # Charger les données de test
    test_data = open_dataset(test_file_path)
    
    # Utiliser les mêmes features que pour l'entraînement
    feature_indices = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # Toutes les matières
    
    # Préparer les données de test
    X_test = []
    valid_indices = []  # Pour garder trace des lignes valides
    
    print("Préparation des données de test...")
    for i in range(1, len(test_data[0])):  # Skip header
        row_features = []
        skip_row = False
        
        for feat_idx in feature_indices:
            if test_data[feat_idx][i] is None or test_data[feat_idx][i] == '':
                skip_row = True
                break
            try:
                row_features.append(float(test_data[feat_idx][i]))
            except ValueError:
                skip_row = True
                break
        
        if not skip_row:
            X_test.append(row_features)
            valid_indices.append(i)
    
    X_test = np.array(X_test)
    print(f"Données de test préparées: {len(X_test)} échantillons valides sur {len(test_data[0])-1}")
    
    # Normaliser avec les paramètres du modèle d'entraînement
    X_test_norm = (X_test - mean) / std
    
    # Faire les prédictions
    predictions, scores = predict(X_test_norm, weights, biases)
    predicted_houses = [houses[pred] for pred in predictions]
    
    # Créer le fichier de résultats
    print(f"Sauvegarde des prédictions dans {output_file}...")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Hogwarts House'])  # Header
        
        prediction_idx = 0
        for i in range(1, len(test_data[0])):  # Pour chaque ligne du fichier original
            if i in valid_indices:
                # On a une prédiction pour cette ligne
                index = test_data[0][i]  # Index original
                house = predicted_houses[prediction_idx]
                writer.writerow([index, house])
                prediction_idx += 1
            else:
                # Ligne avec données manquantes - prédiction par défaut
                index = test_data[0][i]
                writer.writerow([index, "Hufflepuff"])  # Maison par défaut
    
    print(f"Prédictions terminées ! Fichier sauvé: {output_file}")
    
    # Afficher quelques statistiques
    print("\n=== STATISTIQUES DES PRÉDICTIONS ===")
    house_counts = {}
    for house in predicted_houses:
        house_counts[house] = house_counts.get(house, 0) + 1
    
    for house, count in house_counts.items():
        percentage = (count / len(predicted_houses)) * 100
        print(f"{house:12}: {count:4} étudiants ({percentage:.1f}%)")
    
    # Afficher quelques exemples avec confiance
    print("\n=== EXEMPLES DE PRÉDICTIONS ===")
    for i in range(min(10, len(predicted_houses))):
        idx = valid_indices[i]
        student_index = test_data[0][idx]
        first_name = test_data[2][idx] if test_data[2][idx] else "?"
        last_name = test_data[3][idx] if test_data[3][idx] else "?"
        predicted_house = predicted_houses[i]
        confidence = scores[i][predictions[i]]
        
        print(f"Index {student_index}: {first_name} {last_name} → {predicted_house} (confiance: {confidence:.3f})")
    
    return predicted_houses


def train_and_save_model(dataset_path="datasets/dataset_train.csv", model_filename="model.npz"):
    """
    Entraîner un modèle et le sauvegarder
    """
    # Charger et entraîner
    data = open_dataset(dataset_path)
    weights, biases, mean, std, houses, house_to_int, feature_names = train_hogwarts_classifier(data)
    
    # Afficher l'analyse des poids
    display_weights(weights, biases, houses, feature_names)
    
    # Sauvegarder
    save_model(weights, biases, mean, std, houses, house_to_int, feature_names, model_filename)
    
    return weights, biases, mean, std, houses, house_to_int, feature_names


# Test de l'algorithme de régression logistique
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Mode prédiction sur fichier de test
        if sys.argv[1] == "predict":
            if len(sys.argv) < 3:
                print("Usage: python stochastic.py predict <test_file.csv> [output_file.csv]")
                sys.exit(1)
            
            test_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else ".csv"
            
            # Vérifier si le modèle existe
            try:
                predict_test_file(test_file, "model.npz", output_file)
            except FileNotFoundError:
                print("Modèle non trouvé ! Entraînement d'un nouveau modèle...")
                train_and_save_model()
                predict_test_file(test_file, "model.npz", output_file)
        
        elif sys.argv[1] == "train":
            # Mode entraînement seulement
            train_file = sys.argv[2] if len(sys.argv) > 2 else "datasets/dataset_train.csv"
            train_and_save_model(train_file)
        
        elif sys.argv[1] == "weights":
            # Mode affichage des poids
            model_file = sys.argv[2] if len(sys.argv) > 2 else "model.npz"
            try:
                weights, biases, mean, std, houses, house_to_int, feature_names = load_model(model_file)
                display_weights(weights, biases, houses, feature_names)
                
                # Proposer de sauvegarder en format lisible
                save_readable = input("\nVoulez-vous sauvegarder les poids en format lisible ? (o/n): ").lower()
                if save_readable in ['o', 'oui', 'y', 'yes']:
                    readable_file = model_file.replace('.npz', '_weights.txt')
                    save_weights_readable(weights, biases, houses, feature_names, readable_file)
            except FileNotFoundError:
                print(f"Modèle {model_file} non trouvé !")
        
        else:
            print("Commandes disponibles:")
            print("  python stochastic.py train [dataset.csv]     - Entraîner un modèle")
            print("  python stochastic.py predict <test.csv>      - Faire des prédictions") 
            print("  python stochastic.py weights [model.npz]     - Afficher les poids")
            print("  python stochastic.py                         - Mode démonstration")
    
    else:
        # Mode par défaut: entraînement + démonstration
        # Charger les données
        data = open_dataset("datasets/dataset_train.csv")
        
        # Afficher les colonnes pour debug
        print("Colonnes du dataset:")
        for i, col in enumerate(data):
            print(f"{i}: {col[0]}")
        
        # Entraîner le classificateur
        weights, biases, mean, std, houses, house_to_int, feature_names = train_hogwarts_classifier(data)
        
        print("\n=== MODÈLE ENTRAÎNÉ ===")
        
        # Afficher l'analyse détaillée des poids
        display_weights(weights, biases, houses, feature_names)
        
        # Sauvegarder le modèle
        save_model(weights, biases, mean, std, houses, house_to_int, feature_names)
        
        # Test sur le fichier de test s'il existe
        try:
            print("\n" + "="*50)
            predict_test_file("datasets/dataset_test.csv", "model.npz", "houses.csv")
        except FileNotFoundError:
            print("\nFichier de test non trouvé. Pour faire des prédictions:")
            print("python stochastic.py predict <test_file.csv> [output_file.csv]")

