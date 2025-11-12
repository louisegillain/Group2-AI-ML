using UnityEngine;

public class NumberTile : MonoBehaviour
{
    public int NumberValue;
    public Material DefaultMaterial;
    public Material SuccessMaterial;

    private bool m_Visited;
    private MeshRenderer m_Renderer;

    public bool IsVisited => m_Visited;

    public void VisitTile()
    {
        EnsureRenderer();
        m_Renderer.sharedMaterial = SuccessMaterial;
        m_Visited = true;
    }

    public void ResetTile()
    {
        EnsureRenderer();
        m_Renderer.sharedMaterial = DefaultMaterial;
        m_Visited = false;
    }

    private void EnsureRenderer()
    {
        if (m_Renderer != null)
        {
            return;
        }
        m_Renderer = GetComponentInChildren<MeshRenderer>();
    }
}


