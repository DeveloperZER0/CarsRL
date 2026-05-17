using UnityEngine;

public class oponygenerator : MonoBehaviour
{
    [Header("Źródło znaczników")]
    public Transform kontenerZnacznikow;

    [Header("Ustawienia Opon")]
    public GameObject oponaPrefab;
    public float skalaOpony = 100.0f;

    [Header("Ustawienia Przeszkód")]
    public GameObject przeszkodaPrefab;
    public float skalaPrzeszkody = 1.0f;

    [ContextMenu("Wygeneruj Wszystko")]
    public void GenerujWszystko()
    {
        if (kontenerZnacznikow == null)
        {
            Debug.LogError("Brak kontenera znaczników! Przeciągnij LOSTAKIZLASU do pola 'Kontener Znacznikow'.");
            return;
        }

        GameObject stareOpony = GameObject.Find("Opony_Gotowe");
        if (stareOpony != null) DestroyImmediate(stareOpony);

        GameObject starePrzeszkody = GameObject.Find("Przeszkody_Gotowe");
        if (starePrzeszkody != null) DestroyImmediate(starePrzeszkody);

        GameObject folderNaOpony = new GameObject("Opony_Gotowe");
        folderNaOpony.transform.position = transform.position;

        GameObject folderNaPrzeszkody = new GameObject("Przeszkody_Gotowe");
        folderNaPrzeszkody.transform.position = transform.position;

        int licznikOpon = 0;
        int licznikPrzeszkod = 0;

        foreach (Transform znacznik in kontenerZnacznikow)
        {
            string nazwaZnacznika = znacznik.name.ToLower();

            if (nazwaZnacznika.Contains("opon"))
            {
                if (oponaPrefab != null)
                {
                    GameObject nowaOpona = Instantiate(oponaPrefab, znacznik.position, znacznik.rotation);
                    nowaOpona.transform.localScale = Vector3.one * skalaOpony;
                    nowaOpona.transform.SetParent(folderNaOpony.transform);
                    licznikOpon++;
                }
            }
            else if (nazwaZnacznika.Contains("przeszkoda"))
            {
                if (przeszkodaPrefab != null)
                {
                    GameObject nowaPrzeszkoda = Instantiate(przeszkodaPrefab, znacznik.position, znacznik.rotation);
                    nowaPrzeszkoda.transform.localScale = Vector3.one * skalaPrzeszkody;
                    nowaPrzeszkoda.transform.SetParent(folderNaPrzeszkody.transform);
                    licznikPrzeszkod++;
                }
            }
        }

        Debug.Log($"<color=green>Finał!</color> Wygenerowano: <b>{licznikOpon}</b> opon oraz <b>{licznikPrzeszkod}</b> przeszkód.");

        if (licznikOpon == 0 && licznikPrzeszkod == 0)
        {
            Debug.LogWarning("Nie znaleziono żadnych znaczników. Sprawdź czy kontener zawiera obiekty z odpowiednimi nazwami.");
        }
    }
}