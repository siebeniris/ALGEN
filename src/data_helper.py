import os
from datasets import load_dataset


def load_data_for_decoder(lang="eng-literal"):
    lang_data_dir = "/Users/yiyichen/Documents/experiments/datasets/Morphology-Matters-corpus"
    folderpath = lang

    lang_data_dir_ = os.path.join(lang_data_dir, folderpath)
    with open(os.path.join(lang_data_dir_, "train.txt")) as f:
        train_data = [x.replace("\n", "") for x in f.readlines()]

    with open(os.path.join(lang_data_dir_, "test.txt")) as f:
        test_data = [x.replace("\n", "") for x in f.readlines()][:200]
    return train_data, test_data




def load_data(dataset_name, language, nr_samples=500):
    if dataset_name == "flores":
        datadir = "/Users/yiyichen/Documents/experiments/LanguageGraph/data/floresp-v2.0-rc.2/devtest"
        # language: ace_Arab, deu_Latn, jpn_Jpan
        datapath = os.path.join(datadir, f"devtest.{language}")
        with open(datapath, "r") as f:
            datalist = [x.replace("\n", "") for x in f.readlines()]
        if len(datalist) > nr_samples:
            datalist = datalist[:nr_samples]
        else:
            print(f"the whole dataset has {len(datalist)} samples < {nr_samples}")

        return datalist
    elif dataset_name == "glot500":
        # load glot data with 500 languages
        print("loading data from Glot500")
        data_path = "cis-lmu/Glot500"

        dataset = load_dataset(data_path, language, split="train")
        if len(dataset) > nr_samples:
            dataset = dataset.shuffle(seed=42).select(range(nr_samples))

        datalist = dataset["text"]
        return datalist

    elif dataset_name == "Morphology":
        # TODO: refactor here later to have different languages.
        data_path="/Users/yiyichen/Documents/experiments/datasets/Morphology-Matters-corpus/eng-literal/train.txt"
        print(f"Loading data from {data_path}")
        with open(data_path) as f:
            data = [x.replace("\n", "") for x in f.readlines()]
        print(f"Data length {len(data)}")
        # data_sampled = data[:10000]+data[-300:]
        return data[:1000] + data[-200:]

    else:
        print(f"There are 20 sentences of the same meaning in different languages for testing...")
        datalist = ["Hello, world!", "Hallo, Welt!", "你好，世界！",
                    "Hej, verden!", "Hallo, wereld!",
                    "Bonjour, le monde!", "Γεια σου, Κόσμος!", "Здравствуй, мир!", "こんにちは、世界！", "¡Hola, mundo!",
                    "مرحبًا أيها العالم!"]

        lang_and_sentences = [
            {
                "language": "English",
                "sentence": "I believe that education is the most powerful tool we can use to change the world, because it empowers individuals, opens up opportunities, and builds bridges across cultures and generations."
            },
            {
                "language": "Danish",
                "sentence": "Jeg tror, at uddannelse er det mest kraftfulde værktøj, vi kan bruge til at ændre verden, fordi det styrker enkeltpersoner, åbner muligheder og bygger broer mellem kulturer og generationer."
            },
            {
                "language": "French",
                "sentence": "Je crois que l'éducation est l'outil le plus puissant que nous pouvons utiliser pour changer le monde, car elle donne du pouvoir aux individus, ouvre des opportunités et crée des ponts entre les cultures et les générations."
            },
            {
                "language": "Spanish",
                "sentence": "Creo que la educación es la herramienta más poderosa que podemos utilizar para cambiar el mundo, ya que empodera a las personas, abre oportunidades y construye puentes entre culturas y generaciones."
            },
            {
                "language": "German",
                "sentence": "Ich glaube, dass Bildung das mächtigste Werkzeug ist, das wir nutzen können, um die Welt zu verändern, da sie Menschen stärkt, Möglichkeiten eröffnet und Brücken zwischen Kulturen und Generationen baut."
            },
            {
                "language": "Japanese",
                "sentence": "私は、教育が世界を変えるために使用できる最も強力なツールであると信じています。なぜなら、それは個人をエンパワーメントし、機会を開き、文化や世代を超えた橋を築くからです。"
            },
            {
                "language": "Chinese",
                "sentence": "我相信教育是我们可以用来改变世界的最强大的工具，因为它赋予个人权力，打开机遇，并在文化和代际之间架起桥梁。"
            },
            {
                "language": "Hindi",
                "sentence": "मेरा मानना है कि शिक्षा दुनिया को बदलने के लिए सबसे शक्तिशाली उपकरण है, क्योंकि यह व्यक्तियों को सशक्त बनाती है, अवसर प्रदान करती है, और संस्कृतियों और पीढ़ियों के बीच सेतु बनाती है।"
            },
            {
                "language": "Arabic",
                "sentence": "أعتقد أن التعليم هو الأداة الأكثر قوة التي يمكننا استخدامها لتغيير العالم، لأنه يمنح الأفراد القوة، ويفتح الفرص، ويبني جسوراً بين الثقافات والأجيال."
            },
            {
                "language": "Portuguese",
                "sentence": "Acredito que a educação é a ferramenta mais poderosa que podemos usar para mudar o mundo, porque ela empodera os indivíduos, abre oportunidades e constrói pontes entre culturas e gerações."
            },
            {
                "language": "Russian",
                "sentence": "Я верю, что образование — это самый мощный инструмент, который мы можем использовать, чтобы изменить мир, потому что оно дает людям силы, открывает возможности и строит мосты между культурами и поколениями."
            },
            {
                "language": "Italian",
                "sentence": "Credo che l'educazione sia lo strumento più potente che possiamo usare per cambiare il mondo, perché dà potere agli individui, apre opportunità e costruisce ponti tra culture e generazioni."
            },
            {
                "language": "Korean",
                "sentence": "저는 교육이 세상을 바꿀 수 있는 가장 강력한 도구라고 믿습니다. 왜냐하면 그것은 개인에게 힘을 실어주고, 기회를 열어주며, 문화와 세대를 초월한 다리를 놓기 때문입니다."
            },
            {
                "language": "Turkish",
                "sentence": "Eğitimin dünyayı değiştirmek için kullanabileceğimiz en güçlü araç olduğuna inanıyorum, çünkü bireyleri güçlendirir, fırsatlar yaratır ve kültürler ve nesiller arasında köprüler kurar."
            },
            {
                "language": "Dutch",
                "sentence": "Ik geloof dat onderwijs het krachtigste instrument is dat we kunnen gebruiken om de wereld te veranderen, omdat het individuen versterkt, kansen opent en bruggen bouwt tussen culturen en generaties."
            },
            {
                "language": "Swedish",
                "sentence": "Jag tror att utbildning är det mest kraftfulla verktyget vi kan använda för att förändra världen, eftersom det stärker individer, öppnar möjligheter och bygger broar mellan kulturer och generationer."
            },
            {
                "language": "Greek",
                "sentence": "Πιστεύω ότι η εκπαίδευση είναι το πιο ισχυρό εργαλείο που μπορούμε να χρησιμοποιήσουμε για να αλλάξουμε τον κόσμο, επειδή δίνει δύναμη στα άτομα, ανοίγει ευκαιρίες και χτίζει γέφυρες μεταξύ πολιτισμών και γενεών."
            },
            {
                "language": "Polish",
                "sentence": "Wierzę, że edukacja jest najpotężniejszym narzędziem, którego możemy użyć, aby zmienić świat, ponieważ daje ludziom siłę, otwiera możliwości i buduje mosty między kulturami i pokoleniami."
            },
            {
                "language": "Finnish",
                "sentence": "Uskon, että koulutus on voimakkain työkalu, jota voimme käyttää maailman muuttamiseen, sillä se antaa voimaa yksilöille, avaa mahdollisuuksia ja rakentaa siltoja kulttuurien ja sukupolvien välille."
            },
            {
                "language": "Hungarian",
                "sentence": "Úgy hiszem, hogy az oktatás a legerősebb eszköz, amellyel megváltoztathatjuk a világot, mert megerősíti az egyéneket, lehetőségeket nyit meg, és hidakat épít a kultúrák és generációk között."
            }
        ]
        datalist = [x["sentence"] for x in lang_and_sentences]
        return datalist
